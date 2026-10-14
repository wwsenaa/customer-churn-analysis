import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

RANDOM_STATE = 42

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

os.makedirs("dashboard_data", exist_ok=True)

sns.set_theme(style="whitegrid")

df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("\n===== VERİ SETİ =====")
print("İlk boyut:", df.shape)
print("\nİlk 5 satır:")
print(df.head())

print("\nVeri tipleri:")
print(df.dtypes)

print("\nEksik değer sayıları:")
print(df.isnull().sum().sort_values(ascending=False))

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

eksik = df[df["TotalCharges"].isna()]

print("\nEksik TotalCharges satır sayısı:", len(eksik))

if len(eksik) > 0:
    print(
        "Bu satırların tenure değerleri:",
        eksik["tenure"].unique()
    )

    print(
        "Churn dağılımı:\n",
        eksik["Churn"].value_counts()
    )

df = df.dropna(subset=["TotalCharges"]).copy()

print("\nTemizleme sonrası boyut:", df.shape)

print("\nChurn dağılımı:")
print(df["Churn"].value_counts())

print("\nChurn yüzdeleri:")
print(
    (df["Churn"].value_counts(normalize=True) * 100).round(2)
)

print("\nTekrarlanan satır sayısı:", df.duplicated().sum())

print(
    "Tekrarlanan müşteri ID sayısı:",
    df["customerID"].duplicated().sum()
)

conn = sqlite3.connect("churn.db")

df.to_sql(
    "customers",
    conn,
    if_exists="replace",
    index=False
)

sorgular = {
    "Sözleşme": """
        SELECT
            Contract,
            COUNT(*) AS musteri,
            ROUND(
                AVG(CASE WHEN Churn = 'Yes'
                         THEN 1.0 ELSE 0.0 END) * 100,
                2
            ) AS churn_orani
        FROM customers
        GROUP BY Contract
        ORDER BY churn_orani DESC
    """,

    "İnternet servisi": """
        SELECT
            InternetService,
            COUNT(*) AS musteri,
            ROUND(
                AVG(CASE WHEN Churn = 'Yes'
                         THEN 1.0 ELSE 0.0 END) * 100,
                2
            ) AS churn_orani
        FROM customers
        GROUP BY InternetService
        ORDER BY churn_orani DESC
    """,

    "Ödeme yöntemi": """
        SELECT
            PaymentMethod,
            COUNT(*) AS musteri,
            ROUND(
                AVG(CASE WHEN Churn = 'Yes'
                         THEN 1.0 ELSE 0.0 END) * 100,
                2
            ) AS churn_orani
        FROM customers
        GROUP BY PaymentMethod
        ORDER BY churn_orani DESC
    """,

    "Churn durumuna göre ortalamalar": """
        SELECT
            Churn,
            ROUND(AVG(MonthlyCharges), 2) AS ort_aylik_ucret,
            ROUND(AVG(TotalCharges), 2) AS ort_toplam_ucret,
            ROUND(AVG(tenure), 1) AS ort_tenure
        FROM customers
        GROUP BY Churn
    """,

    "Ödeme yöntemi ve sözleşme": """
        SELECT
            Contract,
            PaymentMethod,
            COUNT(*) AS musteri,
            ROUND(
                AVG(CASE WHEN Churn = 'Yes'
                         THEN 1.0 ELSE 0.0 END) * 100,
                2
            ) AS churn_orani
        FROM customers
        GROUP BY Contract, PaymentMethod
        ORDER BY churn_orani DESC
    """
}

sql_sonuclari = {}

for baslik, sorgu in sorgular.items():
    print(f"\n===== SQL: {baslik} =====")

    sonuc = pd.read_sql_query(sorgu, conn)
    sql_sonuclari[baslik] = sonuc

    print(sonuc.to_string(index=False))

conn.close()

df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 6, 12, 24, 48, 72],
    labels=[
        "0-6 months",
        "7-12 months",
        "13-24 months",
        "25-48 months",
        "49-72 months"
    ],
    include_lowest=True
)

df["MonthlyChargesGroup"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 30, 60, 90, 120],
    labels=["0-30", "30-60", "60-90", "90-120"],
    include_lowest=True
)

print("\n===== GENEL CHURN ORANI =====")

churn_orani = df["ChurnFlag"].mean() * 100
print(f"Churn oranı: %{churn_orani:.2f}")

kategorik_sutunlar = [
    "Contract",
    "TenureGroup",
    "InternetService",
    "PaymentMethod"
]

for col in kategorik_sutunlar:

    oran = (
        df.groupby(col, observed=False)["ChurnFlag"]
        .mean()
        .mul(100)
    )

    print(f"\n===== {col} =====")
    print(oran.round(2))

    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=oran.index.astype(str),
        y=oran.values
    )

    plt.title(f"Churn Rate by {col}")
    plt.xlabel(col)
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Customer Tenure by Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.show()


sayisal_sutunlar = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

plt.figure(figsize=(7, 5))

sns.heatmap(
    df[sayisal_sutunlar].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Numerical Feature Correlation")
plt.tight_layout()
plt.show()

df_model = df.drop(
    columns=[
        "customerID",
        "TenureGroup",
        "MonthlyChargesGroup",
        "ChurnFlag"
    ]
).copy()

df_model["Churn"] = df_model["Churn"].map({
    "Yes": 1,
    "No": 0
})

df_model = pd.get_dummies(
    df_model,
    drop_first=True,
    dtype=int
)

X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_train
)

print("\n===== VERİ BÖLÜNÜMÜ =====")
print("Model geliştirme eğitim kümesi:", X_tr.shape)
print("Doğrulama kümesi:", X_val.shape)
print("Nihai eğitim kümesi:", X_train.shape)
print("Test kümesi:", X_test.shape)

f2_scorer = make_scorer(
    fbeta_score,
    beta=2,
    zero_division=0
)

modeller = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=1
    ),

    "XGBoost + SMOTE": ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=1
        ))
    ])
}

cv_sonuclari = []

for isim, model in modeller.items():

    print(f"\nÇapraz doğrulama çalışıyor: {isim}")

    cv_skor = cross_validate(
        model,
        X_tr,
        y_tr,
        cv=cv,
        scoring={
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "f2": f2_scorer,
            "roc_auc": "roc_auc"
        },
        n_jobs=-1,
        error_score="raise"
    )

    satir = {"Model": isim}

    for metrik in [
        "precision",
        "recall",
        "f1",
        "f2",
        "roc_auc"
    ]:
        skorlar = cv_skor[f"test_{metrik}"]

        satir[f"{metrik}_mean"] = skorlar.mean()
        satir[f"{metrik}_std"] = skorlar.std()

    cv_sonuclari.append(satir)

cv_df = pd.DataFrame(cv_sonuclari)

print("\n===== 5 KATLI CV SONUÇLARI =====")
print(cv_df.round(3).to_string(index=False))

best_model_name = (
    cv_df.sort_values("f2_mean", ascending=False)
    .iloc[0]["Model"]
)

best_model_template = modeller[best_model_name]

print("\nSeçilen model:", best_model_name)

val_model = clone(best_model_template)
val_model.fit(X_tr, y_tr)

val_olasilik = val_model.predict_proba(X_val)[:, 1]

esik_satirlari = []

for esik in np.round(np.arange(0.20, 0.81, 0.05), 2):

    val_tahmin = (val_olasilik >= esik).astype(int)

    esik_satirlari.append({
        "Threshold": esik,

        "Precision": precision_score(
            y_val,
            val_tahmin,
            zero_division=0
        ),

        "Recall": recall_score(
            y_val,
            val_tahmin,
            zero_division=0
        ),

        "F1": f1_score(
            y_val,
            val_tahmin,
            zero_division=0
        ),

        "F2": fbeta_score(
            y_val,
            val_tahmin,
            beta=2,
            zero_division=0
        )
    })

esik_df = pd.DataFrame(esik_satirlari)

print("\n===== DOĞRULAMA EŞİK ANALİZİ =====")
print(esik_df.round(3).to_string(index=False))

best_threshold = float(
    esik_df.loc[esik_df["F2"].idxmax(), "Threshold"]
)

print("\nSeçilen model:", best_model_name)
print("Seçilen eşik:", best_threshold)

final_model = clone(best_model_template)
final_model.fit(X_train, y_train)

test_olasilik = final_model.predict_proba(X_test)[:, 1]

test_sonuclari = []

for ad, esik in [
    ("Default 0.50", 0.50),
    ("Selected Threshold", best_threshold)
]:

    test_tahmin = (test_olasilik >= esik).astype(int)

    precision = precision_score(
        y_test,
        test_tahmin,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_tahmin,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_tahmin,
        zero_division=0
    )

    f2 = fbeta_score(
        y_test,
        test_tahmin,
        beta=2,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_test,
        test_tahmin
    )

    roc_auc = roc_auc_score(
        y_test,
        test_olasilik
    )

    print(f"\n===== TEST: {ad} =====")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_tahmin))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        test_tahmin,
        zero_division=0
    ))

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"F2: {f2:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")

    test_sonuclari.append({
        "Model": best_model_name,
        "Threshold_Type": ad,
        "Threshold": esik,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "F2": f2,
        "ROC_AUC": roc_auc
    })

test_metrics_df = pd.DataFrame(test_sonuclari)


selected_test_predictions = (
    test_olasilik >= best_threshold
).astype(int)

cm = confusion_matrix(
    y_test,
    selected_test_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

if isinstance(final_model, (Pipeline, ImbPipeline)):
    fitted_estimator = final_model.named_steps["model"]
else:
    fitted_estimator = final_model

if hasattr(fitted_estimator, "feature_importances_"):

    importance_values = fitted_estimator.feature_importances_
    importance_label = "Feature Importance"

elif hasattr(fitted_estimator, "coef_"):

    importance_values = np.abs(fitted_estimator.coef_[0])
    importance_label = "Absolute Coefficient"

else:
    raise ValueError(
        "Seçilen model için özellik önemleri hesaplanamadı."
    )

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance_values
}).sort_values(
    "Importance",
    ascending=False
)

print("\n===== EN ÖNEMLİ 10 ÖZELLİK =====")
print(importance_df.head(10).to_string(index=False))

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title(f"Top 10 Features - {best_model_name}")
plt.xlabel(importance_label)
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

dashboard_df = df.copy()

dashboard_olasiliklar = cross_val_predict(
    clone(best_model_template),
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

dashboard_df["ChurnRiskScore"] = np.round(
    dashboard_olasiliklar,
    3
)

dashboard_df["PredictedChurn"] = (
    dashboard_df["ChurnRiskScore"] >= best_threshold
).astype(int)

dashboard_df["RiskLevel"] = np.select(
    [
        dashboard_df["ChurnRiskScore"] < 0.30,
        dashboard_df["ChurnRiskScore"] < 0.60
    ],
    [
        "Low",
        "Medium"
    ],
    default="High"
)

dashboard_df["TenureGroup"] = (
    dashboard_df["TenureGroup"].astype(str)
)

dashboard_df["MonthlyChargesGroup"] = (
    dashboard_df["MonthlyChargesGroup"].astype(str)
)

dashboard_df.to_csv(
    "dashboard_data/customer_churn_dashboard.csv",
    index=False
)

cv_df.to_csv(
    "dashboard_data/model_metrics_cv.csv",
    index=False
)

test_metrics_df.to_csv(
    "dashboard_data/model_metrics.csv",
    index=False
)

esik_df.to_csv(
    "dashboard_data/threshold_analysis.csv",
    index=False
)


importance_df.head(10).to_csv(
    "dashboard_data/feature_importance.csv",
    index=False
)

for baslik, sonuc in sql_sonuclari.items():

    dosya_adi = (
        baslik.lower()
        .replace(" ", "_")
        .replace("ı", "i")
        .replace("ş", "s")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ç", "c")
    )

    sonuc.to_csv(
        f"dashboard_data/sql_{dosya_adi}.csv",
        index=False
    )


model_bilgisi = pd.DataFrame([{
    "Selected_Model": best_model_name,
    "Selected_Threshold": best_threshold,
    "Threshold_Metric": "F2",
    "Random_State": RANDOM_STATE,
    "CV_Folds": 5
}])

model_bilgisi.to_csv(
    "dashboard_data/model_selection.csv",
    index=False
)

print("\n===== PROJE TAMAMLANDI =====")
print("Seçilen model:", best_model_name)
print("Seçilen eşik:", best_threshold)

print("\nOluşturulan dosyalar:")

for dosya in sorted(os.listdir("dashboard_data")):
    print("dashboard_data/" + dosya)

print("\nNot:")
print(
    "Test sonuçları nihai model değerlendirmesi içindir. "
    "Power BI risk skorları göreceli risk skorlarıdır; "
    "kalibre edilmiş churn olasılıkları olarak sunulmamalıdır."
)
