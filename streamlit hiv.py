import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


st.set_page_config(
    page_title="HIV Train-Test Split & Geospatial Analysis",
    layout="wide",
)

sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")


@st.cache_data
def load_data():
    df = pd.read_csv("hiv_data_cleaned.csv")
    return df


@st.cache_data
def build_features(df: pd.DataFrame):
    df_features = df.copy()

    # Feature engineering (disederhanakan dari notebook)
    df_features["Case_Range"] = df_features["Count_max"] - df_features["Count_min"]
    df_features["Case_Range_Ratio"] = (
        df_features["Case_Range"] / df_features["Count_median"]
    ).fillna(0)

    df_features["Log_Count_median"] = np.log1p(df_features["Count_median"])
    df_features["Log_Count_min"] = np.log1p(df_features["Count_min"])
    df_features["Log_Count_max"] = np.log1p(df_features["Count_max"])

    df_features["Year_Normalized"] = (
        df_features["Year"] - df_features["Year"].min()
    ) / (df_features["Year"].max() - df_features["Year"].min())

    le_region = LabelEncoder()
    df_features["WHO_Region_Encoded"] = le_region.fit_transform(
        df_features["WHO Region"]
    )

    le_country = LabelEncoder()
    df_features["Country_Encoded"] = le_country.fit_transform(df_features["Country"])

    df_features["Min_Max_Ratio"] = (
        df_features["Count_min"] / df_features["Count_max"]
    ).fillna(0)
    df_features["Range_Median_Ratio"] = (
        df_features["Case_Range"] / df_features["Count_median"]
    ).fillna(0)

    return df_features


@st.cache_data
def split_data(df_features: pd.DataFrame, test_size: float, random_state: int):
    target_col = "Count_median"
    exclude_cols = [
        target_col,
        "Count",
        "Log_Count_median",
        "Country",
        "WHO Region",
    ]

    feature_cols = [
        col
        for col in df_features.columns
        if col not in exclude_cols and df_features[col].dtype in ["int64", "float64"]
    ]

    X = df_features[feature_cols]
    y = df_features[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    return X, y, X_train, X_test, y_train, y_test, feature_cols


def format_large(x):
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def main():
    st.title("📊 HIV Train-Test Split & Geospatial Dashboard")
    st.write(
        "Analisis komprehensif train-test split dan distribusi geografis jumlah orang hidup dengan HIV."
    )
    st.caption(f"📅 Analisis dijalankan pada {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Sidebar controls
    st.sidebar.header("Pengaturan")
    test_size = st.sidebar.slider(
        "Proporsi Test Set", min_value=0.1, max_value=0.4, value=0.2, step=0.05
    )
    random_state = st.sidebar.number_input(
        "Random State", min_value=0, max_value=9999, value=42, step=1
    )
    year_filter = st.sidebar.multiselect(
        "Filter Tahun (opsional)", options=[2000, 2005, 2010, 2018], default=[]
    )

    df = load_data()
    if year_filter:
        df = df[df["Year"].isin(year_filter)]

    st.subheader("📂 Informasi Dataset")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Jumlah Baris", len(df))
    with c2:
        st.metric("Jumlah Kolom", len(df.columns))
    with c3:
        st.metric("Tahun Unik", df["Year"].nunique())

    with st.expander("Lihat preview data"):
        st.dataframe(df.head())

    df_features = build_features(df)
    X, y, X_train, X_test, y_train, y_test, feature_cols = split_data(
        df_features, test_size, random_state
    )

    # ===== Seksi 1: Ringkasan Split =====
    st.subheader("✂️ Train-Test Split Summary")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total Sampel", len(X))
    with col_b:
        st.metric("Train Set", f"{len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    with col_c:
        st.metric("Test Set", f"{len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sizes = [len(X_train), len(X_test)]
        labels = ["Train", "Test"]
        colors = ["#3498db", "#e74c3c"]
        ax.pie(
            sizes,
            labels=[f"{l}\n{v} ({v/len(X)*100:.1f}%)" for l, v in zip(labels, sizes)],
            colors=colors,
            startangle=90,
            explode=(0.05, 0.08),
            textprops={"fontsize": 10, "fontweight": "bold"},
        )
        ax.set_title("Proporsi Train vs Test")
        st.pyplot(fig)

    with col2:
        stats_df = pd.DataFrame(
            {
                "Set": ["Train", "Test"],
                "Mean": [y_train.mean(), y_test.mean()],
                "Median": [y_train.median(), y_test.median()],
                "Std Dev": [y_train.std(), y_test.std()],
                "Min": [y_train.min(), y_test.min()],
                "Max": [y_train.max(), y_test.max()],
            }
        )
        # Format hanya kolom numerik supaya tidak error saat kolom string diformat
        numeric_cols = ["Mean", "Median", "Std Dev", "Min", "Max"]
        st.dataframe(
            stats_df.style.format("{:,.0f}", subset=pd.IndexSlice[:, numeric_cols])
        )

    # ===== Seksi 2: Distribusi Target =====
    st.subheader("📈 Distribusi Target: Train vs Test")
    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots()
        ax.hist(
            y_train,
            bins=30,
            alpha=0.7,
            label="Train",
            color="#3498db",
            edgecolor="black",
        )
        ax.hist(
            y_test,
            bins=30,
            alpha=0.7,
            label="Test",
            color="#e74c3c",
            edgecolor="black",
        )
        ax.axvline(y_train.mean(), color="blue", linestyle="--", label="Train Mean")
        ax.axvline(y_test.mean(), color="red", linestyle="--", label="Test Mean")
        ax.set_xlabel("Count_median")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram Count_median")
        ax.legend()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_large(x)))
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots()
        ax.hist(
            np.log1p(y_train),
            bins=30,
            alpha=0.7,
            label="Train",
            color="#3498db",
            edgecolor="black",
        )
        ax.hist(
            np.log1p(y_test),
            bins=30,
            alpha=0.7,
            label="Test",
            color="#e74c3c",
            edgecolor="black",
        )
        ax.set_xlabel("log(Count_median + 1)")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram (Log Scale)")
        ax.legend()
        st.pyplot(fig)

    col5, col6 = st.columns(2)
    with col5:
        fig, ax = plt.subplots()
        ax.boxplot([y_train, y_test], labels=["Train", "Test"], patch_artist=True)
        ax.set_title("Boxplot Target")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_large(x)))
        st.pyplot(fig)

    with col6:
        fig, ax = plt.subplots()
        parts = ax.violinplot([y_train, y_test], positions=[1, 2], showmeans=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Train", "Test"])
        ax.set_title("Violin Plot Target")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_large(x)))
        st.pyplot(fig)

    # ===== Seksi 3: Geospatial / Choropleth =====
    st.subheader("🗺️ Peta Choropleth Distribusi Kasus HIV per Negara")

    # Siapkan data peta
    train_indices = X_train.index
    test_indices = X_test.index

    df_train_map = df_features.loc[
        train_indices, ["Country", "Count_median", "WHO Region", "Year"]
    ].copy()
    df_train_map["Set"] = "Train"
    df_test_map = df_features.loc[
        test_indices, ["Country", "Count_median", "WHO Region", "Year"]
    ].copy()
    df_test_map["Set"] = "Test"

    df_map = pd.concat([df_train_map, df_test_map], ignore_index=True)
    df_map_agg = (
        df_map.groupby("Country")
        .agg(
            {
                "Count_median": "mean",
                "WHO Region": "first",
                "Set": lambda x: ", ".join(x.unique()),
                "Year": "mean",
            }
        )
        .reset_index()
    )

    if PLOTLY_AVAILABLE:
        country_mapping = {
            "United States of America": "United States",
            "United Republic of Tanzania": "Tanzania",
            "Russian Federation": "Russia",
            "Bolivia (Plurinational State of)": "Bolivia",
            "Venezuela (Bolivarian Republic of)": "Venezuela",
            "Iran (Islamic Republic of)": "Iran",
            "Syrian Arab Republic": "Syria",
            "Republic of Korea": "South Korea",
            "Democratic Republic of the Congo": "Congo, Democratic Republic of the",
            "Republic of the Congo": "Congo",
            "Lao People's Democratic Republic": "Laos",
            "Viet Nam": "Vietnam",
            "The former Yugoslav Republic of Macedonia": "North Macedonia",
            "Republic of Moldova": "Moldova",
            "Czech Republic": "Czechia",
        }

        df_map_plotly = df_map_agg.copy()
        df_map_plotly["Country_Plotly"] = df_map_plotly["Country"].map(
            country_mapping
        ).fillna(df_map_plotly["Country"])
        df_map_plotly["Kasus_Formatted"] = df_map_plotly["Count_median"].apply(
            lambda x: format_large(x)
        )

        # Peta 1: distribusi kasus per negara (sesuai gambar pertama)
        fig_map = px.choropleth(
            df_map_plotly,
            locations="Country_Plotly",
            locationmode="country names",
            color="Count_median",
            hover_name="Country",
            hover_data={
                "WHO Region": True,
                "Count_median": ":,.0f",
                "Kasus_Formatted": True,
                "Set": True,
                "Country_Plotly": False,
            },
            color_continuous_scale="Reds",
            labels={"Count_median": "Jumlah Kasus HIV"},
            title="Peta Choropleth: Distribusi Kasus HIV per Negara<br><sub>Hover untuk melihat: Nama Negara, Region WHO, dan Jumlah Kasus</sub>",
        )

        st.plotly_chart(fig_map, use_container_width=True)

        # Peta 2: 6 WHO Region dengan warna berbeda (sesuai gambar kedua)
        st.subheader("🗺️ Peta Choropleth: 6 WHO Region dengan Warna Berbeda")

        region_colors = {
            "Africa": "#ff6b9d",  # pink
            "Europe": "#daa520",  # golden brown
            "Americas": "#2ecc71",  # green
            "Eastern Mediterranean": "#16a085",  # teal
            "Western Pacific": "#3498db",  # blue
            "South-East Asia": "#9b59b6",  # purple
        }

        df_region_plot = df_map_plotly.copy()
        # total kasus per region untuk tooltip
        region_totals = df_region_plot.groupby("WHO Region")["Count_median"].sum().to_dict()

        customdata = []
        for _, row in df_region_plot.iterrows():
            total_region = region_totals[row["WHO Region"]]
            total_region_fmt = format_large(total_region)
            customdata.append(
                [
                    row["WHO Region"],
                    row["Count_median"],
                    row["Kasus_Formatted"],
                    total_region,
                    total_region_fmt,
                ]
            )

        fig_region = px.choropleth(
            df_region_plot,
            locations="Country_Plotly",
            locationmode="country names",
            color="WHO Region",
            hover_name="Country",
            hover_data={
                "Count_median": ":,.0f",
                "Kasus_Formatted": True,
                "WHO Region": True,
            },
            color_discrete_map=region_colors,
            title=(
                "Peta Choropleth: 6 WHO Region dengan Warna Berbeda<br>"
                "<sub>Setiap region memiliki warna berbeda - hover untuk melihat: "
                "Nama Negara, Region WHO, dan Jumlah Kasus</sub>"
            ),
        )

        fig_region.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "<b>Region WHO:</b> %{customdata[0]}<br>"
                "<b>Kasus Negara Ini:</b> %{customdata[1]:,.0f} (%{customdata[2]})<br>"
                "<b>Total Kasus Region:</b> %{customdata[3]:,.0f} (%{customdata[4]})<br>"
                "<extra></extra>"
            ),
            customdata=customdata,
        )

        st.plotly_chart(fig_region, use_container_width=True)
    else:
        st.info("Plotly tidak tersedia, menampilkan ringkasan per region sebagai gantinya.")
        region_stats = (
            df_map_agg.groupby("WHO Region")["Count_median"]
            .agg(["mean", "sum", "count"])
            .round(2)
        )
        st.dataframe(region_stats)

    # ===== Seksi 4: Heatmap Region vs Tahun =====
    st.subheader("🔥 Heatmap Kasus HIV per WHO Region dan Tahun")
    region_year_data = (
        df_features.groupby(["WHO Region", "Year"])["Count_median"]
        .mean()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        region_year_data,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Rata-rata Kasus HIV"},
        ax=ax,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("WHO Region")
    st.pyplot(fig)

    st.markdown(
        "**Catatan:** Dashboard ini diringkas dari notebook `Progress_Split_Analysis.ipynb` "
        "agar mudah dijalankan di Streamlit dan di-deploy ke Streamlit Community Cloud."
    )


if __name__ == "__main__":
    main()




