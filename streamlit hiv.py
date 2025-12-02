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
    st.markdown("---")

    # Summary Dashboard Interaktif
    st.subheader("📊 Dashboard Ringkasan Eksekutif")
    
    # Hitung metrik kunci
    total_cases_global = df_features["Count_median"].sum()
    avg_cases_per_year = df_features.groupby("Year")["Count_median"].sum().mean()
    years_span = df_features["Year"].max() - df_features["Year"].min() + 1
    total_countries = df_features["Country"].nunique()
    total_regions = df_features["WHO Region"].nunique()
    
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
    
    # Visualisasi ringkasan dengan metrics
    col_dash1, col_dash2, col_dash3, col_dash4 = st.columns(4)
    with col_dash1:
        st.metric("🌍 Total Kasus Global", format_large(total_cases_global), 
                 delta=f"{format_large(last_year_cases - first_year_cases)} dari baseline" if first_year_cases > 0 else None)
    with col_dash2:
        st.metric("📈 Rata-rata/Tahun", format_large(avg_cases_per_year))
    with col_dash3:
        st.metric("📊 Tingkat Pertumbuhan", f"{growth_rate:+.1f}%",
                 delta="Meningkat" if growth_rate > 0 else "Menurun" if growth_rate < 0 else "Stabil")
    with col_dash4:
        st.metric("🗺️ Cakupan", f"{total_countries} negara, {total_regions} region")
    
    # Visualisasi ringkasan eksekutif
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.markdown("**📊 Distribusi Kasus per Tahun**")
        if PLOTLY_AVAILABLE:
            fig_exec = px.bar(
                year_trend_analysis,
                x="Year",
                y="Count_median",
                labels={"Year": "Tahun", "Count_median": "Total Kasus"},
                title="Total Kasus HIV per Tahun",
                color="Count_median",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_exec, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.bar(year_trend_analysis["Year"], year_trend_analysis["Count_median"], color="#e74c3c")
            ax.set_xlabel("Tahun")
            ax.set_ylabel("Total Kasus")
            ax.set_title("Total Kasus HIV per Tahun")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_large(x)))
            st.pyplot(fig)
    
    with col_viz2:
        st.markdown("**📈 Tren Pertumbuhan Global**")
        if PLOTLY_AVAILABLE:
            year_trend_analysis["Growth_Pct"] = year_trend_analysis["Count_median"].pct_change() * 100
            fig_growth = px.line(
                year_trend_analysis,
                x="Year",
                y="Growth_Pct",
                markers=True,
                labels={"Year": "Tahun", "Growth_Pct": "Pertumbuhan (%)"},
                title="Tingkat Pertumbuhan Tahunan (%)"
            )
            fig_growth.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Baseline")
            st.plotly_chart(fig_growth, use_container_width=True)
        else:
            year_trend_analysis["Growth_Pct"] = year_trend_analysis["Count_median"].pct_change() * 100
            fig, ax = plt.subplots()
            ax.plot(year_trend_analysis["Year"], year_trend_analysis["Growth_Pct"], marker="o", color="#3498db")
            ax.axhline(y=0, color="gray", linestyle="--", label="Baseline")
            ax.set_xlabel("Tahun")
            ax.set_ylabel("Pertumbuhan (%)")
            ax.set_title("Tingkat Pertumbuhan Tahunan (%)")
            ax.legend()
            st.pyplot(fig)
    
    st.markdown("---")
    
    # 1. Ringkasan Eksekutif (Detail)
    st.subheader("📊 1. Ringkasan Eksekutif (Detail)")
    
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
                    "Current_Cases": last_val,
                    "First_Cases": first_val
                })
    
    # Inisialisasi variabel untuk menghindari error
    high_risk = pd.DataFrame()
    moderate_risk = pd.DataFrame()
    declining = pd.DataFrame()
    
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
    
    # Matriks Prioritas (Impact vs Growth Rate)
    st.markdown("**📊 Matriks Prioritas Intervensi**")
    if country_growth:
        df_priority = pd.DataFrame(country_growth)
        # Gabungkan dengan total kasus untuk impact
        country_totals = df_features.groupby("Country")["Count_median"].sum().to_dict()
        df_priority["Total_Cases"] = df_priority["Country"].map(country_totals)
        df_priority["Impact_Score"] = (df_priority["Total_Cases"] / df_priority["Total_Cases"].max() * 100).round(1)
        df_priority["Priority_Score"] = (df_priority["Impact_Score"] * 0.6 + abs(df_priority["Growth_Rate"]) * 0.4).round(1)
        df_priority = df_priority.sort_values("Priority_Score", ascending=False)
        
        top_priority = df_priority.head(10)[["Country", "Impact_Score", "Growth_Rate", "Priority_Score", "Current_Cases"]].copy()
        top_priority.columns = ["Negara", "Impact Score", "Growth Rate (%)", "Priority Score", "Kasus Terkini"]
        top_priority["Growth Rate (%)"] = top_priority["Growth Rate (%)"].round(1)
        top_priority["Kasus Terkini"] = top_priority["Kasus Terkini"].apply(format_large)
        
        st.dataframe(top_priority, use_container_width=True, hide_index=True)
        st.caption("**Catatan:** Priority Score = (Impact Score × 0.6) + (|Growth Rate| × 0.4). Semakin tinggi skor, semakin tinggi prioritas intervensi.")

    # 4. Analisis Komparatif & Benchmarking
    st.subheader("📊 4. Analisis Komparatif & Benchmarking")
    
    # Perbandingan region
    region_comparison = (
        df_features.groupby("WHO Region")
        .agg({
            "Count_median": ["sum", "mean", "std"],
            "Country": "nunique"
        })
        .round(2)
    )
    region_comparison.columns = ["Total Kasus", "Rata-rata Kasus", "Std Dev", "Jumlah Negara"]
    region_comparison = region_comparison.sort_values("Total Kasus", ascending=False)
    region_comparison["Total Kasus Formatted"] = region_comparison["Total Kasus"].apply(format_large)
    region_comparison["% dari Global"] = (region_comparison["Total Kasus"] / total_cases_global * 100).round(2)
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.markdown("**📈 Perbandingan WHO Region**")
        region_display = region_comparison[["Total Kasus Formatted", "% dari Global", "Jumlah Negara", "Rata-rata Kasus"]].copy()
        region_display.columns = ["Total Kasus", "% Global", "Jml Negara", "Rata-rata"]
        region_display["Rata-rata"] = region_display["Rata-rata"].apply(format_large)
        st.dataframe(region_display, use_container_width=True)
    
    with col_comp2:
        st.markdown("**📊 Distribusi Kasus per Region**")
        if PLOTLY_AVAILABLE:
            fig_region_pie = px.pie(
                region_comparison.reset_index(),
                values="Total Kasus",
                names="WHO Region",
                title="Proporsi Kasus HIV per WHO Region",
                hole=0.4
            )
            st.plotly_chart(fig_region_pie, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.pie(region_comparison["Total Kasus"], labels=region_comparison.index, autopct='%1.1f%%')
            ax.set_title("Proporsi Kasus HIV per WHO Region")
            st.pyplot(fig)
    
    # 5. Key Performance Indicators (KPIs) & Risk Assessment
    st.subheader("📊 5. Key Performance Indicators (KPIs) & Risk Assessment")
    
    # Hitung KPIs
    top_3_countries_total = top_countries.head(3)["Count_median"].sum()
    top_3_percentage = (top_3_countries_total / total_cases_global * 100).round(1)
    top_5_countries_total = top_countries.head(5)["Count_median"].sum()
    top_5_percentage = (top_5_countries_total / total_cases_global * 100).round(1)
    
    # Risk Score Calculation
    high_risk_count = len(high_risk) if len(high_risk) > 0 else 0
    moderate_risk_count = len(moderate_risk) if len(moderate_risk) > 0 else 0
    
    # Overall Risk Level
    if high_risk_count >= 5:
        overall_risk = "SANGAT TINGGI"
        risk_color = "🔴"
    elif high_risk_count >= 2:
        overall_risk = "TINGGI"
        risk_color = "🟠"
    elif high_risk_count >= 1 or moderate_risk_count >= 5:
        overall_risk = "SEDANG"
        risk_color = "🟡"
    else:
        overall_risk = "RENDAH"
        risk_color = "🟢"
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        st.metric("🎯 Konsentrasi Top 3", f"{top_3_percentage}%", 
                 help="Persentase kasus dari 3 negara teratas")
    with col_kpi2:
        st.metric("🎯 Konsentrasi Top 5", f"{top_5_percentage}%",
                 help="Persentase kasus dari 5 negara teratas")
    with col_kpi3:
        st.metric("⚠️ Negara Risiko Tinggi", f"{high_risk_count}",
                 help="Jumlah negara dengan pertumbuhan >20%")
    with col_kpi4:
        st.metric(f"{risk_color} Overall Risk Level", overall_risk,
                 help="Tingkat risiko keseluruhan berdasarkan analisis data")
    
    # Risk Assessment Matrix
    st.markdown("**📊 Risk Assessment Matrix**")
    
    if country_growth:
        # Buat df_growth untuk Risk Matrix (gunakan yang sudah ada atau buat baru)
        df_risk_matrix = pd.DataFrame(country_growth)
        country_totals = df_features.groupby("Country")["Count_median"].sum().to_dict()
        df_risk_matrix["Total_Cases"] = df_risk_matrix["Country"].map(country_totals)
        
        # Normalize untuk matrix
        max_cases = df_risk_matrix["Total_Cases"].max()
        max_growth = df_risk_matrix["Growth_Rate"].abs().max()
        
        if max_cases > 0 and max_growth > 0:
            df_risk_matrix["Impact_Normalized"] = (df_risk_matrix["Total_Cases"] / max_cases * 100).round(1)
            df_risk_matrix["Growth_Normalized"] = (df_risk_matrix["Growth_Rate"].abs() / max_growth * 100).round(1)
        
        # Visualisasi Risk Matrix
        if max_cases > 0 and max_growth > 0:
            if PLOTLY_AVAILABLE:
                fig_risk = px.scatter(
                    df_risk_matrix.head(20),  # Top 20 untuk readability
                    x="Impact_Normalized",
                    y="Growth_Normalized",
                    size="Total_Cases",
                    color="Growth_Rate",
                    hover_name="Country",
                    hover_data={
                        "Total_Cases": ":,.0f",
                        "Growth_Rate": ":.1f",
                        "Impact_Normalized": ":.1f",
                        "Growth_Normalized": ":.1f"
                    },
                    labels={
                        "Impact_Normalized": "Impact Score (Berdasarkan Total Kasus)",
                        "Growth_Normalized": "Growth Risk Score (Berdasarkan Tingkat Pertumbuhan)",
                        "Growth_Rate": "Growth Rate (%)"
                    },
                    title="Risk Assessment Matrix: Impact vs Growth Risk",
                    color_continuous_scale="RdYlGn_r"
                )
                fig_risk.update_layout(
                    xaxis_title="Impact Score (Semakin tinggi = lebih banyak kasus)",
                    yaxis_title="Growth Risk Score (Semakin tinggi = pertumbuhan lebih cepat)"
                )
                st.plotly_chart(fig_risk, use_container_width=True)
                st.caption("**Interpretasi:** Posisi di kanan atas = risiko tertinggi (banyak kasus + pertumbuhan cepat). Posisi di kiri bawah = risiko lebih rendah.")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    df_risk_matrix.head(20)["Impact_Normalized"],
                    df_risk_matrix.head(20)["Growth_Normalized"],
                    s=df_risk_matrix.head(20)["Total_Cases"] / 1000,
                    c=df_risk_matrix.head(20)["Growth_Rate"],
                    cmap="RdYlGn_r",
                    alpha=0.6
                )
                ax.set_xlabel("Impact Score (Berdasarkan Total Kasus)")
                ax.set_ylabel("Growth Risk Score (Berdasarkan Tingkat Pertumbuhan)")
                ax.set_title("Risk Assessment Matrix: Impact vs Growth Risk")
                plt.colorbar(scatter, ax=ax, label="Growth Rate (%)")
                st.pyplot(fig)
        else:
            st.info("Data tidak cukup untuk membuat Risk Assessment Matrix")
    
    st.markdown("---")
    
    # 6. Sistem Pendukung Keputusan - Rekomendasi Kebijakan
    st.subheader("💡 6. Sistem Pendukung Keputusan: Rekomendasi Kebijakan")
    
    # Generate rekomendasi berdasarkan analisis
    recommendations = []
    
    # Rekomendasi untuk top countries
    for idx, row in top_countries.head(5).iterrows():
        country = row["Country"]
        cases = row["Count_median"]
        # Cek apakah negara ini juga high growth
        is_high_growth = False
        if country_growth:
            country_growth_df = pd.DataFrame(country_growth)
            country_info = country_growth_df[country_growth_df["Country"] == country]
            if len(country_info) > 0 and country_info.iloc[0]["Growth_Rate"] > 20:
                is_high_growth = True
        
        priority = "SANGAT TINGGI" if is_high_growth else "TINGGI"
        recommendations.append({
            "Prioritas": priority,
            "Target": f"Negara: {country}",
            "Alasan": f"Total kasus {format_large(cases)} - termasuk 5 negara dengan beban tertinggi" + (f" dengan pertumbuhan sangat cepat" if is_high_growth else ""),
            "Rekomendasi": f"1. **Intensifikasi Program Pencegahan**: Perluas cakupan program pencegahan dan pengobatan di {country}\n2. **Alokasi Sumber Daya Prioritas**: Fokuskan anggaran untuk testing, treatment, dan care services\n3. **Program Edukasi & Awareness**: Kampanye edukasi yang lebih agresif dan terukur\n4. **Monitoring & Evaluasi**: Sistem tracking real-time untuk deteksi dini peningkatan kasus\n5. **Kolaborasi Internasional**: Kerjasama dengan organisasi internasional untuk best practices",
            "Timeline": "Segera (0-3 bulan)",
            "Budget_Estimate": "Tinggi"
        })
    
    # Rekomendasi untuk high growth countries
    if len(high_risk) > 0:
        for idx, row in high_risk.head(3).iterrows():
            country = row["Country"]
            growth = row["Growth_Rate"]
            current = row["Current_Cases"]
            recommendations.append({
                "Prioritas": "SANGAT TINGGI",
                "Target": f"Negara: {country}",
                "Alasan": f"Pertumbuhan {growth:.1f}% - pertumbuhan sangat cepat dengan {format_large(current)} kasus terkini",
                "Rekomendasi": f"1. **Investigasi Mendalam**: Analisis penyebab peningkatan kasus di {country}\n2. **Program Pencegahan Darurat**: Implementasi intervensi cepat dan terukur\n3. **Monitoring Intensif**: Tracking harian/mingguan untuk deteksi tren\n4. **Rapid Response Team**: Tim khusus untuk penanganan darurat\n5. **Resource Mobilization**: Mobilisasi sumber daya tambahan dari donor internasional",
                "Timeline": "Sangat Segera (0-1 bulan)",
                "Budget_Estimate": "Sangat Tinggi"
            })
    
    # Rekomendasi untuk top regions
    for idx, row in top_regions.head(3).iterrows():
        region = row["WHO Region"]
        pct = row["Persentase"]
        total_region = row["Count_median"]
        recommendations.append({
            "Prioritas": "TINGGI",
            "Target": f"Region: {region}",
            "Alasan": f"Mencakup {pct}% dari total kasus global ({format_large(total_region)} kasus)",
            "Rekomendasi": f"1. **Koordinasi Regional**: Platform koordinasi untuk program pencegahan di {region}\n2. **Sharing Best Practices**: Pertukaran pengalaman antar negara dalam region\n3. **Alokasi Dana Regional**: Pooling resources untuk intervensi terpadu\n4. **Capacity Building**: Pelatihan dan penguatan kapasitas negara-negara dalam region\n5. **Regional Strategy**: Strategi regional yang terintegrasi dan terukur",
            "Timeline": "Jangka Menengah (3-6 bulan)",
            "Budget_Estimate": "Sangat Tinggi"
        })
    
    # Rekomendasi untuk declining countries (best practices)
    if len(declining) > 0:
        for idx, row in declining.head(2).iterrows():
            country = row["Country"]
            growth = row["Growth_Rate"]
            recommendations.append({
                "Prioritas": "INFORMASI",
                "Target": f"Negara: {country} (Best Practice)",
                "Alasan": f"Tren menurun {growth:.1f}% - sukses dalam penanganan HIV/AIDS",
                "Rekomendasi": f"1. **Dokumentasi Best Practices**: Dokumentasi program yang berhasil di {country}\n2. **Knowledge Sharing**: Sharing pengalaman ke negara lain\n3. **Sustaining Success**: Mempertahankan momentum penurunan kasus\n4. **Replication**: Replikasi program ke negara dengan karakteristik serupa",
                "Timeline": "Berkelanjutan",
                "Budget_Estimate": "Sedang"
            })

    if recommendations:
        st.markdown("**🎯 Rekomendasi Kebijakan Berbasis Data (Diurutkan berdasarkan Prioritas):**")
        
        # Filter berdasarkan prioritas
        priority_filter = st.multiselect(
            "Filter berdasarkan Prioritas:",
            options=["SANGAT TINGGI", "TINGGI", "INFORMASI"],
            default=["SANGAT TINGGI", "TINGGI"]
        )
        
        filtered_recs = [r for r in recommendations if r["Prioritas"] in priority_filter]
        
        for i, rec in enumerate(filtered_recs[:10], 1):  # Tampilkan max 10 rekomendasi
            priority_color = {
                "SANGAT TINGGI": "🔴",
                "TINGGI": "🟡",
                "INFORMASI": "🟢"
            }.get(rec["Prioritas"], "⚪")
            
            with st.expander(f"**{priority_color} Rekomendasi #{i} - Prioritas: {rec['Prioritas']}** | Target: {rec['Target']}", expanded=(i <= 2)):
                col_rec1, col_rec2 = st.columns(2)
                with col_rec1:
                    st.markdown(f"**📋 Alasan:** {rec['Alasan']}")
                    st.markdown(f"**⏰ Timeline:** {rec['Timeline']}")
                with col_rec2:
                    st.markdown(f"**💰 Estimasi Budget:** {rec['Budget_Estimate']}")
                
                st.markdown("**💡 Rekomendasi Kebijakan:**")
                st.markdown(rec['Rekomendasi'])

    # 7. Analisis Prediktif Sederhana & Proyeksi
    st.subheader("🔮 7. Analisis Prediktif & Proyeksi")
    
    # Hitung CAGR (Compound Annual Growth Rate)
    if len(year_trend_analysis) > 1:
        years_diff = year_trend_analysis.iloc[-1]["Year"] - year_trend_analysis.iloc[0]["Year"]
        if years_diff > 0 and first_year_cases > 0:
            cagr = (((last_year_cases / first_year_cases) ** (1 / years_diff)) - 1) * 100
        else:
            cagr = 0
    else:
        cagr = 0
    
    # Proyeksi sederhana untuk 3 tahun ke depan (jika tren linear)
    if len(year_trend_analysis) > 1:
        # Hitung rata-rata pertumbuhan per tahun
        year_trend_analysis["Year_Diff"] = year_trend_analysis["Year"].diff()
        year_trend_analysis["Case_Diff"] = year_trend_analysis["Count_median"].diff()
        year_trend_analysis["Yearly_Growth"] = year_trend_analysis["Case_Diff"] / year_trend_analysis["Year_Diff"]
        avg_yearly_growth = year_trend_analysis["Yearly_Growth"].mean()
        
        # Proyeksi 3 tahun ke depan
        last_year = int(year_trend_analysis.iloc[-1]["Year"])
        projection_years = [last_year + 1, last_year + 2, last_year + 3]
        projections = []
        current_cases = last_year_cases
        
        for proj_year in projection_years:
            current_cases += avg_yearly_growth
            projections.append({
                "Tahun": proj_year,
                "Proyeksi Kasus": max(0, current_cases),
                "Pertumbuhan": avg_yearly_growth
            })
        
        df_projection = pd.DataFrame(projections)
        df_projection["Proyeksi Kasus Formatted"] = df_projection["Proyeksi Kasus"].apply(format_large)
    else:
        df_projection = pd.DataFrame()
        avg_yearly_growth = 0
    
    col_proj1, col_proj2 = st.columns(2)
    
    with col_proj1:
        st.markdown("**📊 Metrik Pertumbuhan:**")
        st.metric("CAGR (Compound Annual Growth Rate)", f"{cagr:+.2f}%")
        st.metric("Rata-rata Pertumbuhan per Tahun", format_large(avg_yearly_growth))
        st.metric("Kasus Tahun Terakhir", format_large(last_year_cases))
    
    with col_proj2:
        st.markdown("**🔮 Proyeksi 3 Tahun Ke Depan:**")
        if len(df_projection) > 0:
            proj_display = df_projection[["Tahun", "Proyeksi Kasus Formatted"]].copy()
            proj_display.columns = ["Tahun", "Proyeksi Total Kasus"]
            st.dataframe(proj_display, use_container_width=True, hide_index=True)
            st.caption("**Catatan:** Proyeksi berdasarkan tren linear dari data historis. Aktual dapat berbeda tergantung intervensi.")
        else:
            st.info("Data tidak cukup untuk proyeksi")
    
    # Visualisasi proyeksi
    if len(df_projection) > 0 and PLOTLY_AVAILABLE:
        # Gabungkan data historis dengan proyeksi
        historical_data = year_trend_analysis[["Year", "Count_median"]].copy()
        historical_data["Type"] = "Historis"
        
        projection_data = df_projection[["Tahun", "Proyeksi Kasus"]].copy()
        projection_data.columns = ["Year", "Count_median"]
        projection_data["Type"] = "Proyeksi"
        
        combined_data = pd.concat([historical_data, projection_data], ignore_index=True)
        
        fig_projection = px.line(
            combined_data,
            x="Year",
            y="Count_median",
            color="Type",
            markers=True,
            labels={
                "Year": "Tahun",
                "Count_median": "Total Kasus HIV",
                "Type": "Tipe Data"
            },
            title="Tren Historis & Proyeksi Kasus HIV (3 Tahun Ke Depan)"
        )
        fig_projection.update_traces(line=dict(dash="dash"), selector={"name": "Proyeksi"})
        st.plotly_chart(fig_projection, use_container_width=True)
    
    # 8. Actionable Insights & Next Steps
    st.subheader("🚀 8. Actionable Insights & Langkah Selanjutnya")
    
    # Hitung statistik tambahan untuk insights
    top_3_countries_total = top_countries.head(3)["Count_median"].sum()
    top_3_percentage = (top_3_countries_total / total_cases_global * 100).round(1)
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("**📌 Insight Kunci:**")
        st.markdown(f"""
        - **Fokus Geografis**: {len(top_regions)} region menyumbang sebagian besar kasus global
        - **Konsentrasi Kasus**: Top 3 negara menyumbang {top_3_percentage}% dari total kasus global
        - **Tren Global**: Pertumbuhan kasus **{"meningkat" if growth_rate > 0 else "menurun"}** sebesar {abs(growth_rate):.1f}% dari {int(df_features['Year'].min())} ke {int(df_features['Year'].max())}
        - **Prioritas Intervensi**: {len(high_risk) if len(high_risk) > 0 else 0} negara dengan pertumbuhan >20% memerlukan perhatian segera
        - **CAGR**: {cagr:+.2f}% per tahun menunjukkan tren {"naik" if cagr > 0 else "turun"} yang {"mengkhawatirkan" if cagr > 5 else "stabil"}
        """)
    
    with insights_col2:
        st.markdown("**✅ Langkah Selanjutnya (Action Plan):**")
        st.markdown("""
        **Fase 1 - Immediate (0-1 bulan):**
        1. ✅ **Emergency Response**: Aktivasi rapid response untuk negara risiko tinggi
        2. ✅ **Resource Mobilization**: Mobilisasi sumber daya untuk area prioritas
        
        **Fase 2 - Short Term (1-3 bulan):**
        3. ✅ **Program Intensifikasi**: Implementasi program pencegahan di top 10 negara
        4. ✅ **Monitoring System**: Setup sistem monitoring real-time
        5. ✅ **Stakeholder Engagement**: Koordinasi dengan pemerintah dan organisasi internasional
        
        **Fase 3 - Medium Term (3-6 bulan):**
        6. ✅ **Regional Coordination**: Platform koordinasi regional untuk top 3 region
        7. ✅ **Capacity Building**: Pelatihan dan penguatan kapasitas
        8. ✅ **Best Practice Replication**: Replikasi program sukses ke negara lain
        
        **Fase 4 - Long Term (6-12 bulan):**
        9. ✅ **Evaluation & Optimization**: Evaluasi program dan optimasi berdasarkan hasil
        10. ✅ **Sustainability Planning**: Perencanaan keberlanjutan program
        """)

    # 9. Ekspor Laporan & Dokumentasi
    st.subheader("📥 9. Ekspor Laporan & Dokumentasi")
    
    # Buat laporan lengkap
    report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Siapkan data untuk laporan
    top_5_countries_str = "\n".join([
        f"  {i+1}. {row['Country']}: {format_large(row['Count_median'])} kasus" 
        for i, (_, row) in enumerate(top_countries.head(5).iterrows())
    ])
    
    top_3_regions_str = "\n".join([
        f"  {i+1}. {row['WHO Region']}: {format_large(row['Count_median'])} kasus ({row['Persentase']}%)" 
        for i, (_, row) in enumerate(top_regions.head(3).iterrows())
    ])
    
    recommendations_str = "\n".join([
        f"\n{i+1}. {rec['Prioritas']} - {rec['Target']}\n   Alasan: {rec['Alasan']}\n   Timeline: {rec['Timeline']}\n   Rekomendasi:\n   {rec['Rekomendasi'].replace(chr(10), chr(10) + '   ')}"
        for i, rec in enumerate(recommendations[:8], 1)
    ])
    
    high_risk_str = ""
    if len(high_risk) > 0:
        high_risk_str = "\n".join([
            f"  - {row['Country']}: Pertumbuhan {row['Growth_Rate']:.1f}%, Kasus {format_large(row['Current_Cases'])}"
            for _, row in high_risk.iterrows()
        ])
    else:
        high_risk_str = "  - Tidak ada negara dengan pertumbuhan >20%"
    
    report_summary = f"""
================================================================================
LAPORAN ANALITIK HIV/AIDS - DECISION SUPPORT SYSTEM
Sistem Pendukung Keputusan untuk Rekomendasi Kebijakan Pencegahan HIV/AIDS
================================================================================
Tanggal Laporan: {report_date}
Periode Data: {int(df_features['Year'].min())} - {int(df_features['Year'].max())}
================================================================================

1. RINGKASAN EKSEKUTIF
--------------------------------------------------------------------------------
Total Kasus Global: {format_large(total_cases_global)}
Rata-rata Kasus per Tahun: {format_large(avg_cases_per_year)}
Tingkat Pertumbuhan: {growth_rate:+.1f}%
CAGR (Compound Annual Growth Rate): {cagr:+.2f}%
Rentang Tahun: {int(df_features['Year'].min())} - {int(df_features['Year'].max())}
Jumlah Negara: {df_features['Country'].nunique()}
Jumlah WHO Region: {df_features['WHO Region'].nunique()}

2. IDENTIFIKASI AREA PRIORITAS
--------------------------------------------------------------------------------
TOP 5 NEGARA DENGAN KASUS TERTINGGI:
{top_5_countries_str}

TOP 3 WHO REGION DENGAN KASUS TERTINGGI:
{top_3_regions_str}

3. ANALISIS RISIKO
--------------------------------------------------------------------------------
NEGARA DENGAN RISIKO TINGGI (Pertumbuhan >20%):
{high_risk_str}

4. REKOMENDASI KEBIJAKAN
--------------------------------------------------------------------------------
{recommendations_str}

5. PROYEKSI & TREN
--------------------------------------------------------------------------------
Proyeksi 3 Tahun Ke Depan (berdasarkan tren linear):
"""
    
    if len(df_projection) > 0:
        for _, row in df_projection.iterrows():
            report_summary += f"  - Tahun {int(row['Tahun'])}: {row['Proyeksi Kasus Formatted']} kasus\n"
    else:
        report_summary += "  - Data tidak cukup untuk proyeksi\n"
    
    report_summary += f"""
6. ACTIONABLE INSIGHTS
--------------------------------------------------------------------------------
- Fokus Geografis: {len(top_regions)} region menyumbang sebagian besar kasus global
- Konsentrasi Kasus: Top 3 negara menyumbang {top_3_percentage}% dari total kasus
- Tren Global: Pertumbuhan kasus {"meningkat" if growth_rate > 0 else "menurun"} sebesar {abs(growth_rate):.1f}%
- Prioritas Intervensi: {len(high_risk) if len(high_risk) > 0 else 0} negara memerlukan perhatian segera

7. LANGKAH SELANJUTNYA
--------------------------------------------------------------------------------
Fase 1 - Immediate (0-1 bulan):
  - Emergency Response untuk negara risiko tinggi
  - Resource Mobilization untuk area prioritas

Fase 2 - Short Term (1-3 bulan):
  - Intensifikasi program pencegahan di top 10 negara
  - Setup sistem monitoring real-time
  - Koordinasi dengan stakeholders

Fase 3 - Medium Term (3-6 bulan):
  - Platform koordinasi regional
  - Capacity building dan pelatihan
  - Replikasi best practices

Fase 4 - Long Term (6-12 bulan):
  - Evaluasi dan optimasi program
  - Perencanaan keberlanjutan

================================================================================
Laporan ini dihasilkan secara otomatis oleh Decision Support System
untuk mendukung pengambilan keputusan berbasis data.
================================================================================
"""
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        st.download_button(
            label="📄 Download Laporan Lengkap (TXT)",
            data=report_summary,
            file_name=f"HIV_AIDS_Decision_Support_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col_export2:
        # Ekspor data prioritas ke CSV
        if len(top_countries) > 0:
            priority_data = top_countries.head(10)[["Country", "Count_median"]].copy()
            priority_data.columns = ["Country", "Total_Cases"]
            csv_data = priority_data.to_csv(index=False)
            st.download_button(
                label="📊 Download Data Prioritas (CSV)",
                data=csv_data,
                file_name=f"HIV_Priority_Countries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    st.info("💡 **Tip**: Gunakan laporan ini untuk presentasi kepada stakeholders, perencanaan anggaran, dan dokumentasi keputusan kebijakan.")

    st.markdown(
        "**Catatan:** Dashboard ini diringkas dari notebook `Progress_Split_Analysis.ipynb` "
        "agar mudah dijalankan di Streamlit dan di-deploy ke Streamlit Community Cloud."
    )


if __name__ == "__main__":
    main()


