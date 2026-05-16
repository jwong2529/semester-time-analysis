import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from datetime import date

HOLIDAYS = [
    (date(2025, 11, 3), date(2025, 11, 4)), # academic holiday + election day
    (date(2025, 11, 26), date(2025, 11, 28)), # thanksgiving
]

def is_holiday(d):
    for start, end in HOLIDAYS:
        if start <= d <= end:
            return True
    return False

def compute_crunch_weeks(weekly_totals, z_threshold=1.5):
    # Identifies crunch weeks using z-scores. 
    # A crunch week is defined as one where: z-score > z_threshold
  
    std = weekly_totals.std()

    # guard against division by zero
    if std == 0 or len(weekly_totals) < 2:
        z_scores = pd.Series(0, index=weekly_totals.index)
        crunch_weeks = pd.Series(dtype=float)
        return z_scores, crunch_weeks

    z_scores = (weekly_totals - weekly_totals.mean()) / std
    crunch_weeks = z_scores[z_scores > z_threshold]

    return z_scores, crunch_weeks

def generate_insights(df_all, df_stats, weekly_totals_stats, weekly_pivot_stats, category_stats):
    insights = []

    total_hours = df_all["Hours spent"].sum()
    avg_weekly = weekly_totals_stats.mean()

    insights.append(
        f"You logged **{total_hours:.1f} total hours** this semester, "
        f"averaging **{avg_weekly:.1f} hours per week**."
    )

    # busiest & lightest weeks (using typical weeks for stats)
    busiest_week = weekly_totals_stats.idxmax()
    lightest_week = weekly_totals_stats.idxmin()

    insights.append(
        f"Your busiest typical week was **the week of {busiest_week.strftime("%b %d")}**, "
        f"with **{weekly_totals_stats[busiest_week]:.1f} hours** logged."
    )

    insights.append(
        f"Your lightest typical week was **the week of {lightest_week.strftime("%b %d")}**, "
        f"with **{weekly_totals_stats[lightest_week]:.1f} hours**."
    )

    # category dominance
    category_totals = df_all.groupby("Category")["Hours spent"].sum()
    top_category = category_totals.idxmax()
    top_category_pct = category_totals.max() / total_hours * 100 if total_hours > 0 else 0

    insights.append(
        f"Your most time-consuming category overall was **{top_category}**, "
        f"accounting for **{top_category_pct:.1f}%** of your total time."
    )

    # consistency
    most_consistent = category_stats["consistency_score"].idxmin()
    insights.append(
        f"You were most consistent in **{most_consistent}**, "
        f"showing relatively steady weekly effort."
    )

    # most intense category-week (using stats weeks to avoid finals skew)
    max_idx = weekly_pivot_stats.stack().idxmax()
    max_val = weekly_pivot_stats.stack().max()

    insights.append(
        f"Your most intense typical week-category combo was "
        f"**{max_idx[1]}** during the week of **{max_idx[0].strftime("%b %d")}**, "
        f"with **{max_val:.1f} hours**."
    )

    # weekend behavior
    weekend_hours = df_all[df_all["Is Weekend"]]["Hours spent"].sum()
    weekend_pct = weekend_hours / total_hours * 100 if total_hours > 0 else 0

    insights.append(
        f"Approximately **{weekend_pct:.1f}%** of your total work time (all weeks) happened on weekends."
    )

    # crunch weeks 
    _, crunch_weeks = compute_crunch_weeks(weekly_totals_stats)

    if not crunch_weeks.empty:
        insights.append(
            f"You experienced **{len(crunch_weeks)} crunch week(s)** where "
            f"your workload was significantly higher than normal."
        )
    else:
        insights.append(
            "You did not experience any extreme crunch weeks — your workload was relatively balanced."
        )

    # trend over time
    if len(weekly_totals_stats) < 2:
        slope = 0
    else:
        week_numbers = np.arange(len(weekly_totals_stats))
        slope = np.polyfit(week_numbers, weekly_totals_stats.values, 1)[0]

    if slope > 0.5:
        insights.append("Your workload **increased over the semester**, suggesting rising intensity toward the end.")
    elif slope < -0.5:
        insights.append("Your workload **decreased over the semester**, possibly indicating front-loaded effort.")
    else:
        insights.append("Your workload remained **fairly stable throughout the semester**.")

    # Consistency insight (uses full df)
    daily_logged = df_all.groupby(df_all["Date"].dt.date)["Hours spent"].sum()
    total_days = (df_all["Date"].max() - df_all["Date"].min()).days + 1
    consistency = (len(daily_logged) / total_days) * 100
    insights.append(f"Overall, you logged hours on **{len(daily_logged)} out of {total_days} days** ({consistency:.1f}% consistency).")

    return insights

def load_and_clean(csv_file):
    df = pd.read_csv(csv_file)

    df = df.drop(columns='Category name')
    
    # date parsing for Notion exports
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="mixed",
        errors="coerce"
    )

    df = df.dropna(subset=["Date"])
    df["Hours spent"] = pd.to_numeric(df["Hours spent"], errors="coerce").fillna(0)

    # Monday → Sunday weeks (starts on Monday)
    df["Week"] = df["Date"].dt.to_period("W-SUN").apply(lambda r: r.start_time)

    df["Day of Week"] = df["Date"].dt.day_name()
    df["Is Weekend"] = df["Date"].dt.weekday >= 5
    df["Is Holiday"] = df["Date"].dt.date.apply(is_holiday)

    return df

st.set_page_config(page_title="Semester Analysis", layout="wide")

st.sidebar.title("Data Source")
data_source = st.sidebar.radio(
    "Choose data source",
    ["Upload CSV", "View Sample File"]
)

target_file = None
file_name_display = ""

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your Notion CSV", type="csv")
    if uploaded_file:
        target_file = uploaded_file
        file_name_display = uploaded_file.name
else:
    logs_dir = "logs"
    if os.path.exists(logs_dir):
        sample_files = [f for f in os.listdir(logs_dir) if f.endswith(".csv")]
        if sample_files:
            selected_sample = st.sidebar.selectbox("Select sample file", sample_files)
            if selected_sample:
                target_file = os.path.join(logs_dir, selected_sample)
                file_name_display = selected_sample
        else:
            st.sidebar.info("No sample files found in 'logs/' folder")
    else:
        st.sidebar.error("'logs/' folder not found")

if target_file:
    df = load_and_clean(target_file)
    st.title(f"{file_name_display.replace('.csv', '')}")

    # sidebar filters
    st.sidebar.header("Filters")

    categories = st.sidebar.multiselect(
        "Category",
        sorted(df["Category"].dropna().unique())
    )

    tags = st.sidebar.multiselect(
        "Tag",
        sorted(df["Tags"].dropna().unique())
    )

    # Determine default date range from data
    data_min_date = df["Date"].min().date()
    data_max_date = df["Date"].max().date()

    date_range = st.sidebar.date_input(
        "Date range",
        [data_min_date, data_max_date]
    )

    exclude_holidays = st.sidebar.checkbox("Exclude holidays", value=False)

    min_weekly_hours = st.sidebar.slider(
        "Min hours to include week in averages",
        min_value=0,
        max_value=40,
        value=0,
        help="Exclude weeks where you logged fewer than this many hours (e.g. spring break)."
    )

    available_weeks = sorted(df["Week"].unique())
    weeks_to_exclude = st.sidebar.multiselect(
        "Exclude specific weeks",
        available_weeks,
        format_func=lambda x: x.strftime("%b %d"),
        help="Manually select weeks to remove from the analysis."
    )

    # apply filters
    if categories:
        df = df[df["Category"].isin(categories)]
    if tags:
        df = df[df["Tags"].isin(tags)]

    if len(date_range) == 2:
        df = df[
            (df["Date"].dt.date >= date_range[0]) &
            (df["Date"].dt.date <= date_range[1])
        ]

    if exclude_holidays:
        df = df[~df["Is Holiday"]]

    # Identify excluded weeks for stats, but keep them in df for charts
    weeks_to_exclude_final = list(weeks_to_exclude)
    if min_weekly_hours > 0:
        temp_weekly = df.groupby("Week")["Hours spent"].sum()
        low_activity_weeks = temp_weekly[temp_weekly < min_weekly_hours].index
        weeks_to_exclude_final.extend(low_activity_weeks)
    
    weeks_to_exclude_final = list(set(weeks_to_exclude_final))

    # tables
    weekly_category = (
        df.groupby(["Week", "Category"])["Hours spent"]
        .sum()
        .reset_index()
    )

    weekly_pivot = weekly_category.pivot(
        index="Week",
        columns="Category",
        values="Hours spent"
    ).fillna(0)

    # All-time totals for charts
    weekly_totals_all = weekly_pivot.sum(axis=1)

    # Filtered totals for stats/averages
    weekly_pivot_stats = weekly_pivot[~weekly_pivot.index.isin(weeks_to_exclude_final)]
    
    if weekly_pivot_stats.empty or len(weekly_pivot_stats.columns) == 0:
        st.warning("No data found for the selected filters and typical weeks. Please adjust your filters.")
        st.stop()

    weekly_totals_stats = weekly_pivot_stats.sum(axis=1)
    
    category_stats = weekly_pivot_stats.describe().T
    category_stats["avg_per_week"] = weekly_pivot_stats.mean()
    category_stats["max_week"] = weekly_pivot_stats.max()
    category_stats["std_dev"] = weekly_pivot_stats.std()
    category_stats["consistency_score"] = (
        category_stats["std_dev"] / category_stats["avg_per_week"]
    )

    # tabs
    tabs = st.tabs([
        "Report",
        "Overview",
        "Weekly",
        "Daily Patterns",
        "Weekly × Category",
        "Insights",
        "Categories",
        # "Holidays",
        "Raw Data"
    ])

    # report
    with tabs[0]:
        st.subheader("Auto-Generated Report")

        insights = generate_insights(
            df, # df is currently the base-filtered one (all weeks)
            df[~df["Week"].isin(weeks_to_exclude_final)], # filtered for stats
            weekly_totals_stats,
            weekly_pivot_stats,
            category_stats
        )

        for insight in insights:
            st.markdown(f"- {insight}")

    # overview
    with tabs[1]:
        total_hours = df["Hours spent"].sum()
        avg_weekly = weekly_totals_stats.mean()
        
        # Consistency and Streaks
        daily_hours = df.groupby("Date")["Hours spent"].sum().sort_index()
        all_dates = pd.date_range(start=df["Date"].min(), end=df["Date"].max())
        daily_series = daily_hours.reindex(all_dates, fill_value=0)
        
        # Calculate streak
        is_working = daily_series > 0
        streaks = is_working.groupby((is_working != is_working.shift()).cumsum()).cumsum()
        max_streak = streaks.max()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Hours (All)", f"{total_hours:.1f}")
        c2.metric("Avg Hours / Typical Week", f"{avg_weekly:.1f}")
        c3.metric("Typical Weeks", len(weekly_totals_stats))
        c4.metric("Longest Streak", f"{max_streak} days")

        st.divider()
        
        # Daily Intensity Heatmap (GitHub style)
        st.subheader("Daily Activity Pulse")
        daily_df = daily_series.reset_index()
        daily_df.columns = ["Date", "Hours"]
        daily_df["Week"] = daily_df["Date"].dt.isocalendar().week
        daily_df["Day"] = daily_df["Date"].dt.day_name()
        
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heat_data = daily_df.pivot(index="Day", columns="Week", values="Hours").reindex(day_order)
        
        fig_heat = px.imshow(
            heat_data,
            color_continuous_scale="Viridis",
            labels=dict(x="Week of Year", y="Day of Week", color="Hours"),
            title="Daily Intensity (Hours per Day)"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.divider()

        cum = df.sort_values("Date")
        cum["Cumulative Hours"] = cum["Hours spent"].cumsum()

        fig = px.line(
            cum,
            x="Date",
            y="Cumulative Hours",
            title="Cumulative Hours Over Semester"
        )
        st.plotly_chart(fig, use_container_width=True)

    # weekly
    with tabs[2]:
        fig = px.bar(
            weekly_totals_all.reset_index(),
            x="Week",
            y=0,
            labels={"0": "Hours"},
            title="Total Hours per Week (Including Atypical Weeks)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # daily patterns
    with tabs[3]:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily = (
            df.groupby("Day of Week")["Hours spent"]
            .sum()
            .reindex(order)
            .reset_index()
        )

        fig = px.bar(
            daily,
            x="Day of Week",
            y="Hours spent",
            title="Hours by Day of Week"
        )
        st.plotly_chart(fig, use_container_width=True)

        daily_total_hours = df.groupby(df["Date"].dt.date)["Hours spent"].sum().reset_index()
        daily_total_hours["Day of Week"] = pd.to_datetime(daily_total_hours["Date"]).dt.day_name()

        average_daily_total_hours = (
            daily_total_hours.groupby("Day of Week")["Hours spent"]
            .mean()
            .reindex(order)
            .reset_index()
        )

        fig_avg_daily = px.bar(
            average_daily_total_hours,
            x="Day of Week",
            y="Hours spent",
            title="Average Total Hours Per Weekday"
        )
        st.plotly_chart(fig_avg_daily, use_container_width=True)

        st.divider()
        st.subheader("Weekday Category Focus")
        st.write("Which categories do you typically work on for each day of the week?")
        
        day_cat = df.groupby(["Day of Week", "Category"])["Hours spent"].mean().reset_index()
        fig_day_cat = px.bar(
            day_cat,
            x="Day of Week",
            y="Hours spent",
            color="Category",
            category_orders={"Day of Week": order},
            title="Average Hours per Category by Weekday",
            barmode="stack"
        )
        st.plotly_chart(fig_day_cat, use_container_width=True)

    # weekly x category
    with tabs[4]:
        st.subheader("Category Trends (Week by Week)")

        fig_trend = px.line(
            weekly_category,
            x="Week",
            y="Hours spent",
            color="Category",
            markers=True,
            title="Weekly Hours Trend per Category"
        )

        fig_trend.update_layout(hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)
        st.divider()

        fig = px.bar(
            weekly_category,
            x="Week",
            y="Hours spent",
            color="Category",
            title="Weekly Hours by Category"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.imshow(
            weekly_pivot.T,
            aspect="auto",
            labels=dict(color="Hours"),
            title="Heatmap: Category × Week"
        )
        st.plotly_chart(fig, use_container_width=True)

    # insights
    with tabs[5]:
        busiest_week = weekly_totals_stats.idxmax()
        lightest_week = weekly_totals_stats.idxmin()

        c1, c2 = st.columns(2)
        c1.metric(
            "Busiest Typical Week",
            busiest_week.strftime("%b %d"),
            f"{weekly_totals_stats[busiest_week]:.1f} hrs"
        )
        c2.metric(
            "Lightest Typical Week",
            lightest_week.strftime("%b %d"),
            f"{weekly_totals_stats[lightest_week]:.1f} hrs"
        )

        st.subheader("Category Weekly Stats")

        display_stats = category_stats[
            ["avg_per_week", "max_week", "std_dev", "consistency_score"]
        ].sort_values("avg_per_week", ascending=False)

        st.dataframe(display_stats.style.format("{:.2f}"))

        # most intense category-week ever (including atypical)
        max_idx = weekly_pivot.stack().idxmax()
        max_val = weekly_pivot.stack().max()

        st.success(
            f"Most intense week ever: **{max_idx[1]}** — "
            f"{max_val:.1f} hrs (week of {max_idx[0].strftime("%b %d")})"
        )

        # crunch weeks 
        z_scores, crunch = compute_crunch_weeks(weekly_totals_stats)

        if not crunch.empty:
            st.warning("Crunch Weeks Detected")
            for wk, score in crunch.items():
                st.write(
                    f"- Week of {wk.strftime("%b %d")}: "
                    f"{weekly_totals[wk]:.1f} hrs (z={score:.2f})"
                )
        else:
            st.info("No extreme crunch weeks detected")

    # categories
    with tabs[6]:
        col1, col2 = st.columns([1, 1])
        
        cat = df.groupby("Category")["Hours spent"].sum().reset_index()

        with col1:
            fig = px.pie(
                cat,
                names="Category",
                values="Hours spent",
                title="Time Distribution by Category"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Cumulative Growth by Category
            st.write("**Cumulative Category Growth**")
            # Create a full date range to ensure smooth lines
            date_range_full = pd.date_range(df["Date"].min(), df["Date"].max())
            
            # Pivot, cumsum, and reindex
            cat_pivot = df.pivot_table(
                index="Date", 
                columns="Category", 
                values="Hours spent", 
                aggfunc="sum"
            ).fillna(0)
            
            cat_cum = cat_pivot.cumsum().reindex(date_range_full).ffill().fillna(0)
            
            fig_area = px.area(
                cat_cum,
                labels={"value": "Cumulative Hours", "index": "Date"},
                title="Category Time Investment Over Time"
            )
            st.plotly_chart(fig_area, use_container_width=True)

    # holidays
    # with tabs[7]:
    #     h = (
    #         df.groupby("Is Holiday")["Hours spent"]
    #         .sum()
    #         .reset_index()
    #     )
    #     h["Is Holiday"] = h["Is Holiday"].map({True: "Holiday", False: "Normal Day"})

    #     fig = px.bar(
    #         h,
    #         x="Is Holiday",
    #         y="Hours spent",
    #         title="Holiday vs Normal Day Effort"
    #     )
    #     st.plotly_chart(fig, use_container_width=True)

    # raw data
    with tabs[7]:
        st.dataframe(df.sort_values("Date"))

else:
    st.info("Upload or select a data source to begin")
