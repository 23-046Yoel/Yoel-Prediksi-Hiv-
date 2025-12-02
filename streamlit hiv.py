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

    # ===== Seksi 3: Tren Pertumbuhan & Penyebaran per Tahun =====
    st.subheader("📅 Tren Pertumbuhan & Penyebaran HIV per Tahun")

    # Agregasi total kasus per tahun (semua negara)
    year_trend = (
        df_features.groupby("Year")["Count_median"]
        .sum()
        .reset_index()
        .sort_values("Year")
    )

    # Agregasi per region & tahun
    region_year_trend = (
        df_features.groupby(["WHO Region", "Year"])["Count_median"]
        .sum()
        .reset_index()
        .sort_values("Year")
    )

    col_trend1, col_trend2 = st.columns(2)

    with col_trend1:
        st.markdown("**Total Kasus HIV Global per Tahun**")
        if PLOTLY_AVAILABLE:
            fig_year = px.line(
                year_trend,
                x="Year",
                y="Count_median",
                markers=True,
                labels={
                    "Year": "Tahun",
                    "Count_median": "Total Kasus HIV (Count_median)",
                },
                title="Tren Pertumbuhan Kasus HIV Global",
            )
            fig_year.update_traces(line=dict(color="#e74c3c", width=3))
            st.plotly_chart(fig_year, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.plot(
                year_trend["Year"],
                year_trend["Count_median"],
                marker="o",
                color="#e74c3c",
            )
            ax.set_xlabel("Tahun")
            ax.set_ylabel("Total Kasus HIV (Count_median)")
            ax.set_title("Tren Pertumbuhan Kasus HIV Global")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_large(x)))
            st.pyplot(fig)

    with col_trend2:
        st.markdown("**Tren Kasus HIV per WHO Region**")
        selected_regions = st.multiselect(
            "Pilih WHO Region untuk ditampilkan:",
            options=sorted(region_year_trend["WHO Region"].unique()),
            default=sorted(region_year_trend["WHO Region"].unique()),
        )

        filtered_region_year = region_year_trend[
            region_year_trend["WHO Region"].isin(selected_regions)
        ]

        if PLOTLY_AVAILABLE:
            fig_region_trend = px.line(
                filtered_region_year,
                x="Year",
                y="Count_median",
                color="WHO Region",
                markers=True,
                labels={
                    "Year": "Tahun",
                    "Count_median": "Total Kasus HIV (Count_median)",
                    "WHO Region": "Region WHO",
                },
                title="Tren Penyebaran Kasus HIV per WHO Region",
            )
            st.plotly_chart(fig_region_trend, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            for region, sub_df in filtered_region_year.groupby("WHO Region"):
                ax.plot(
                    sub_df["Year"],
                    sub_df["Count_median"],
                    marker="o",
                    label=region,
                )
            ax.set_xlabel("Tahun")
            ax.set_ylabel("Total Kasus HIV (Count_median)")
            ax.set_title("Tren Penyebaran Kasus HIV per WHO Region")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_large(x)))
            ax.legend()
            st.pyplot(fig)

    # ===== Tren Pertumbuhan per Negara =====
    st.subheader("🌍 Tren Pertumbuhan Kasus HIV per Negara")

    # Agregasi per negara & tahun
    country_year_trend = (
        df_features.groupby(["Country", "Year"])["Count_median"]
        .sum()
        .reset_index()
        .sort_values("Year")
    )

    # Pilih WHO Region (opsional) untuk membatasi daftar negara
    col_country1, col_country2 = st.columns(2)

    with col_country1:
        available_regions = sorted(df_features["WHO Region"].unique())
        selected_region_filter = st.selectbox(
            "Filter Negara berdasarkan WHO Region (opsional):",
            options=["Semua Region"] + available_regions,
            index=0,
        )

    # Filter negara sesuai region bila dipilih
    if selected_region_filter != "Semua Region":
        available_countries = sorted(
            df_features.loc[
                df_features["WHO Region"] == selected_region_filter, "Country"
            ].unique()
        )
    else:
        available_countries = sorted(df_features["Country"].unique())

    with col_country2:
        default_countries = available_countries[:5]
        selected_countries = st.multiselect(
            "Pilih 1–5 negara untuk dianalisis:",
            options=available_countries,
            default=default_countries,
        )

    filtered_country_year = country_year_trend[
        country_year_trend["Country"].isin(selected_countries)
    ]

    if selected_countries:
        if PLOTLY_AVAILABLE:
            fig_country_trend = px.line(
                filtered_country_year,
                x="Year",
                y="Count_median",
                color="Country",
                markers=True,
                labels={
                    "Year": "Tahun",
                    "Count_median": "Total Kasus HIV (Count_median)",
                    "Country": "Negara",
                },
                title="Tren Pertumbuhan Kasus HIV per Negara Terpilih",
            )
            st.plotly_chart(fig_country_trend, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            for country, sub_df in filtered_country_year.groupby("Country"):
                ax.plot(
                    sub_df["Year"],
                    sub_df["Count_median"],
                    marker="o",
                    label=country,
                )
            ax.set_xlabel("Tahun")
            ax.set_ylabel("Total Kasus HIV (Count_median)")
            ax.set_title("Tren Pertumbuhan Kasus HIV per Negara Terpilih")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_large(x)))
            ax.legend()
            st.pyplot(fig)
    else:
        st.info("Pilih minimal satu negara untuk melihat tren pertumbuhan.")

    # ===== Seksi 4: Geospatial / Choropleth =====
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

    # ===== Seksi 5: Output & Decision Support =====
    st.header("📋 Output & Decision Support")
    st.markdown(
        "**Laporan Analitik dan Sistem Pendukung Keputusan untuk Rekomendasi Kebijakan Pencegahan HIV/AIDS**"
    )

    # Analisis data untuk laporan
    # 1. Ringkasan Eksekutif
    st.subheader("📊 1. Ringkasan Eksekutif")
    
    # Hitung metrik kunci
    total_cases_global = df_features["Count_median"].sum()
    avg_cases_per_year = df_features.groupby("Year")["Count_median"].sum().mean()
    years_span = df_features["Year"].max() - df_features["Year"].min() + 1
    
    # Analisis pertumbuhan
    year_trend_analysis = (
        df_features.groupby("Year")["Count_median"]
        .sum()
        .reset_index()
        .sort_values("Year")
    )
    if len(year_trend_analysis) > 1:
        first_year_cases = year_trend_analysis.iloc[0]["Count_median"]
        last_year_cases = year_trend_analysis.iloc[-1]["Count_median"]
        growth_rate = ((last_year_cases - first_year_cases) / first_year_cases) * 100 if first_year_cases > 0 else 0
    else:
        growth_rate = 0
        first_year_cases = last_year_cases = 0

    col_exec1, col_exec2, col_exec3, col_exec4 = st.columns(4)
    with col_exec1:
        st.metric("Total Kasus Global", format_large(total_cases_global))
    with col_exec2:
        st.metric("Rata-rata Kasus/Tahun", format_large(avg_cases_per_year))
    with col_exec3:
        st.metric("Tingkat Pertumbuhan", f"{growth_rate:+.1f}%")
    with col_exec4:
        st.metric("Rentang Tahun", f"{int(df_features['Year'].min())}-{int(df_features['Year'].max())}")

    # 2. Identifikasi Area Prioritas
    st.subheader("🎯 2. Identifikasi Area Prioritas")
    
    # Top 10 negara dengan kasus tertinggi
    top_countries = (
        df_features.groupby("Country")["Count_median"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_countries["Kasus_Formatted"] = top_countries["Count_median"].apply(format_large)
    
    # Top 5 region dengan kasus tertinggi
    top_regions = (
        df_features.groupby("WHO Region")["Count_median"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    top_regions["Kasus_Formatted"] = top_regions["Count_median"].apply(format_large)
    top_regions["Persentase"] = (top_regions["Count_median"] / total_cases_global * 100).round(2)

    col_priority1, col_priority2 = st.columns(2)
    
    with col_priority1:
        st.markdown("**🏆 Top 10 Negara dengan Kasus Tertinggi**")
        top_countries_display = top_countries[["Country", "Kasus_Formatted", "Count_median"]].copy()
        top_countries_display.columns = ["Negara", "Total Kasus", "Nilai"]
        top_countries_display = top_countries_display.drop(columns=["Nilai"])
        st.dataframe(top_countries_display, use_container_width=True, hide_index=True)
    
    with col_priority2:
        st.markdown("**🌍 Top 5 WHO Region dengan Kasus Tertinggi**")
        top_regions_display = top_regions[["WHO Region", "Kasus_Formatted", "Persentase"]].copy()
        top_regions_display.columns = ["Region WHO", "Total Kasus", "Persentase (%)"]
        st.dataframe(top_regions_display, use_container_width=True, hide_index=True)

    # 3. Analisis Tren & Risiko
    st.subheader("📈 3. Analisis Tren & Penilaian Risiko")
    
    # Identifikasi negara dengan pertumbuhan tercepat
    country_growth = []
    for country in df_features["Country"].unique():
        country_data = df_features[df_features["Country"] == country].groupby("Year")["Count_median"].sum().sort_index()
        if len(country_data) > 1:
            first_val = country_data.iloc[0]
            last_val = country_data.iloc[-1]
            if first_val > 0:
                growth_pct = ((last_val - first_val) / first_val) * 100
                country_growth.append({
                    "Country": country,
                    "Growth_Rate": growth_pct,
                    "First_Year": country_data.index[0],
                    "Last_Year": country_data.index[-1],
                    "Current_Cases": last_val
                })
    
    if country_growth:
        df_growth = pd.DataFrame(country_growth)
        df_growth = df_growth.sort_values("Growth_Rate", ascending=False)
        
        # Kategorisasi risiko
        high_risk = df_growth[df_growth["Growth_Rate"] > 20].head(5)
        moderate_risk = df_growth[(df_growth["Growth_Rate"] > 0) & (df_growth["Growth_Rate"] <= 20)].head(5)
        declining = df_growth[df_growth["Growth_Rate"] < 0].head(5)

        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        with col_risk1:
            st.markdown("**🔴 Risiko Tinggi (Pertumbuhan >20%)**")
            if len(high_risk) > 0:
                high_risk_display = high_risk[["Country", "Growth_Rate", "Current_Cases"]].copy()
                high_risk_display.columns = ["Negara", "Pertumbuhan (%)", "Kasus Terkini"]
                high_risk_display["Pertumbuhan (%)"] = high_risk_display["Pertumbuhan (%)"].round(1)
                high_risk_display["Kasus Terkini"] = high_risk_display["Kasus Terkini"].apply(format_large)
                st.dataframe(high_risk_display, use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada negara dengan pertumbuhan >20%")
        
        with col_risk2:
            st.markdown("**🟡 Risiko Sedang (Pertumbuhan 0-20%)**")
            if len(moderate_risk) > 0:
                moderate_risk_display = moderate_risk[["Country", "Growth_Rate", "Current_Cases"]].copy()
                moderate_risk_display.columns = ["Negara", "Pertumbuhan (%)", "Kasus Terkini"]
                moderate_risk_display["Pertumbuhan (%)"] = moderate_risk_display["Pertumbuhan (%)"].round(1)
                moderate_risk_display["Kasus Terkini"] = moderate_risk_display["Kasus Terkini"].apply(format_large)
                st.dataframe(moderate_risk_display, use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada data")
        
        with col_risk3:
            st.markdown("**🟢 Tren Menurun (Negatif)**")
            if len(declining) > 0:
                declining_display = declining[["Country", "Growth_Rate", "Current_Cases"]].copy()
                declining_display.columns = ["Negara", "Pertumbuhan (%)", "Kasus Terkini"]
                declining_display["Pertumbuhan (%)"] = declining_display["Pertumbuhan (%)"].round(1)
                declining_display["Kasus Terkini"] = declining_display["Kasus Terkini"].apply(format_large)
                st.dataframe(declining_display, use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada negara dengan tren menurun")

    # 4. Sistem Pendukung Keputusan - Rekomendasi Kebijakan
    st.subheader("💡 4. Sistem Pendukung Keputusan: Rekomendasi Kebijakan")
    
    # Generate rekomendasi berdasarkan analisis
    recommendations = []
    
    # Rekomendasi untuk top countries
    for idx, row in top_countries.head(5).iterrows():
        country = row["Country"]
        cases = row["Count_median"]
        recommendations.append({
            "Prioritas": "SANGAT TINGGI",
            "Target": f"Negara: {country}",
            "Alasan": f"Total kasus {format_large(cases)} - termasuk 5 negara dengan beban tertinggi",
            "Rekomendasi": f"1. Intensifikasi program pencegahan dan pengobatan di {country}\n2. Alokasi sumber daya prioritas untuk testing dan treatment\n3. Program edukasi dan awareness yang lebih agresif"
        })
    
    # Rekomendasi untuk high growth countries
    if len(high_risk) > 0:
        for idx, row in high_risk.head(3).iterrows():
            country = row["Country"]
            growth = row["Growth_Rate"]
            recommendations.append({
                "Prioritas": "TINGGI",
                "Target": f"Negara: {country}",
                "Alasan": f"Pertumbuhan {growth:.1f}% - pertumbuhan sangat cepat",
                "Rekomendasi": f"1. Investigasi penyebab peningkatan kasus di {country}\n2. Implementasi program pencegahan darurat\n3. Monitoring dan evaluasi intensif"
            })
    
    # Rekomendasi untuk top regions
    for idx, row in top_regions.head(3).iterrows():
        region = row["WHO Region"]
        pct = row["Persentase"]
        recommendations.append({
            "Prioritas": "TINGGI",
            "Target": f"Region: {region}",
            "Alasan": f"Mencakup {pct}% dari total kasus global",
            "Rekomendasi": f"1. Koordinasi regional untuk program pencegahan di {region}\n2. Sharing best practices antar negara dalam region\n3. Alokasi dana regional untuk intervensi terpadu"
        })

    if recommendations:
        st.markdown("**Rekomendasi Kebijakan Berbasis Data:**")
        
        for i, rec in enumerate(recommendations[:8], 1):  # Tampilkan max 8 rekomendasi
            with st.expander(f"**Rekomendasi #{i} - Prioritas: {rec['Prioritas']}** | Target: {rec['Target']}", expanded=(i <= 3)):
                st.markdown(f"**Alasan:** {rec['Alasan']}")
                st.markdown(f"**Rekomendasi Kebijakan:**")
                st.markdown(rec['Rekomendasi'])

    # 5. Actionable Insights & Next Steps
    st.subheader("🚀 5. Actionable Insights & Langkah Selanjutnya")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("**📌 Insight Kunci:**")
        st.markdown("""
        - **Fokus Geografis**: {} region menyumbang sebagian besar kasus global
        - **Tren Global**: Pertumbuhan kasus {} dari {} ke {}
        - **Prioritas Intervensi**: {} negara memerlukan perhatian segera
        """.format(
            len(top_regions),
            "meningkat" if growth_rate > 0 else "menurun",
            int(df_features['Year'].min()),
            int(df_features['Year'].max()),
            len(high_risk) if len(high_risk) > 0 else 0
        ))
    
    with insights_col2:
        st.markdown("**✅ Langkah Selanjutnya:**")
        st.markdown("""
        1. **Monitoring Berkelanjutan**: Tracking kasus per kuartal untuk deteksi dini
        2. **Evaluasi Program**: Review efektivitas program pencegahan di area prioritas
        3. **Alokasi Sumber Daya**: Fokuskan anggaran ke negara/region dengan beban tertinggi
        4. **Kolaborasi Internasional**: Koordinasi antar negara untuk best practices
        5. **Data-Driven Decision**: Gunakan dashboard ini untuk evaluasi berkala
        """)

    # 6. Download Laporan
    st.subheader("📥 6. Ekspor Laporan")
    
    # Buat summary report
    report_summary = f"""
# LAPORAN ANALITIK HIV/AIDS - DECISION SUPPORT SYSTEM
## Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### RINGKASAN EKSEKUTIF
- Total Kasus Global: {format_large(total_cases_global)}
- Rata-rata Kasus per Tahun: {format_large(avg_cases_per_year)}
- Tingkat Pertumbuhan: {growth_rate:+.1f}%
- Rentang Tahun: {int(df_features['Year'].min())}-{int(df_features['Year'].max())}

### TOP 5 NEGARA PRIORITAS
{chr(10).join([f"{i+1}. {row['Country']}: {format_large(row['Count_median'])} kasus" for i, (_, row) in enumerate(top_countries.head(5).iterrows())])}

### TOP 3 REGION PRIORITAS
{chr(10).join([f"{i+1}. {row['WHO Region']}: {format_large(row['Count_median'])} kasus ({row['Persentase']}%)" for i, (_, row) in enumerate(top_regions.head(3).iterrows())])}

### REKOMENDASI KEBIJAKAN
{chr(10).join([f"{i+1}. {rec['Target']}: {rec['Alasan']}" for i, rec in enumerate(recommendations[:5], 1)])}
"""
    
    st.download_button(
        label="📄 Download Laporan Ringkas (TXT)",
        data=report_summary,
        file_name=f"HIV_AIDS_Decision_Support_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

    st.markdown(
        "**Catatan:** Dashboard ini diringkas dari notebook `Progress_Split_Analysis.ipynb` "
        "agar mudah dijalankan di Streamlit dan di-deploy ke Streamlit Community Cloud."
    )


if __name__ == "__main__":
    main()


