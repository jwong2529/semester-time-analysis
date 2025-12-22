import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import date

SEMESTER_START = date(2025, 9, 2)
SEMESTER_END = date(2025, 12, 18)

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

def generate_insights(df, weekly_totals, weekly_pivot, category_stats):
    insights = []

    total_hours = df["Hours spent"].sum()
    avg_weekly = weekly_totals.mean()

    insights.append(
        f"You logged **{total_hours:.1f} total hours** this semester, "
        f"averaging **{avg_weekly:.1f} hours per week**."
    )

    # busiest & lightest weeks
    busiest_week = weekly_totals.idxmax()
    lightest_week = weekly_totals.idxmin()

    insights.append(
        f"Your busiest week was **the week of {busiest_week.strftime('%b %d')}**, "
        f"with **{weekly_totals[busiest_week]:.1f} hours** logged."
    )

    insights.append(
        f"Your lightest week was **the week of {lightest_week.strftime('%b %d')}**, "
        f"with **{weekly_totals[lightest_week]:.1f} hours**."
    )

    # category dominance
    category_totals = df.groupby("Category")["Hours spent"].sum()
    top_category = category_totals.idxmax()
    top_category_pct = category_totals.max() / total_hours * 100

    insights.append(
        f"Your most time-consuming category was **{top_category}**, "
        f"accounting for **{top_category_pct:.1f}%** of your total time."
    )

    # consistency
    most_consistent = category_stats["consistency_score"].idxmin()
    insights.append(
        f"You were most consistent in **{most_consistent}**, "
        f"showing relatively steady weekly effort."
    )

    # most intense category-week
    max_idx = weekly_pivot.stack().idxmax()
    max_val = weekly_pivot.stack().max()

    insights.append(
        f"The most intense single week-category combination was "
        f"**{max_idx[1]}** during the week of **{max_idx[0].strftime('%b %d')}**, "
        f"with **{max_val:.1f} hours**."
    )

    # weekend behavior
    weekend_hours = df[df["Is Weekend"]]["Hours spent"].sum()
    weekend_pct = weekend_hours / total_hours * 100 if total_hours > 0 else 0

    insights.append(
        f"Approximately **{weekend_pct:.1f}%** of your total work time happened on weekends."
    )

    # crunch weeks 
    _, crunch_weeks = compute_crunch_weeks(weekly_totals)

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
    week_numbers = np.arange(len(weekly_totals))
    slope = np.polyfit(week_numbers, weekly_totals.values, 1)[0]

    if slope > 0.5:
        insights.append("Your workload **increased over the semester**, suggesting rising intensity toward the end.")
    elif slope < -0.5:
        insights.append("Your workload **decreased over the semester**, possibly indicating front-loaded effort.")
    else:
        insights.append("Your workload remained **fairly stable throughout the semester**.")

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

    # semester filter
    df = df[
        (df["Date"].dt.date >= SEMESTER_START) &
        (df["Date"].dt.date <= SEMESTER_END)
    ]

    # monday → sunday weeks
    df["Week"] = df["Date"].dt.to_period("W-MON").apply(lambda r: r.start_time)

    df["Day of Week"] = df["Date"].dt.day_name()
    df["Is Weekend"] = df["Date"].dt.weekday >= 5
    df["Is Holiday"] = df["Date"].dt.date.apply(is_holiday)

    return df

st.set_page_config(page_title="Semester Analysis", layout="wide")

uploaded_file = st.sidebar.file_uploader("Upload your Notion CSV", type="csv")

if uploaded_file:
    df = load_and_clean(uploaded_file)
    st.title(f"{uploaded_file.name.replace('.csv', '')}")

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

    date_range = st.sidebar.date_input(
        "Date range",
        [SEMESTER_START, SEMESTER_END]
    )

    exclude_holidays = st.sidebar.checkbox("Exclude holidays", value=False)

    # apply filters
    if categories:
        df = df[df["Category"].isin(categories)]
    if tags:
        df = df[df["Tags"].isin(tags)]

    df = df[
        (df["Date"].dt.date >= date_range[0]) &
        (df["Date"].dt.date <= date_range[1])
    ]

    if exclude_holidays:
        df = df[~df["Is Holiday"]]

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

    weekly_totals = weekly_pivot.sum(axis=1)

    category_stats = weekly_pivot.describe().T
    category_stats["avg_per_week"] = weekly_pivot.mean()
    category_stats["max_week"] = weekly_pivot.max()
    category_stats["std_dev"] = weekly_pivot.std()
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
        "Holidays",
        "Raw Data"
    ])

    # report
    with tabs[0]:
        st.subheader("Auto-Generated Report")

        insights = generate_insights(
            df,
            weekly_totals,
            weekly_pivot,
            category_stats
        )

        for insight in insights:
            st.markdown(f"- {insight}")

    # overview
    with tabs[1]:
        total_hours = df["Hours spent"].sum()
        avg_weekly = weekly_totals.mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Hours", f"{total_hours:.1f}")
        c2.metric("Avg Hours / Week", f"{avg_weekly:.1f}")
        c3.metric("Total Weeks", len(weekly_totals))

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
            weekly_totals.reset_index(),
            x="Week",
            y=0,
            labels={"0": "Hours"},
            title="Total Hours per Week"
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

    # weekly x category
    with tabs[4]:
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
        busiest_week = weekly_totals.idxmax()
        lightest_week = weekly_totals.idxmin()

        c1, c2 = st.columns(2)
        c1.metric(
            "Busiest Week",
            busiest_week.strftime("%b %d"),
            f"{weekly_totals[busiest_week]:.1f} hrs"
        )
        c2.metric(
            "Lightest Week",
            lightest_week.strftime("%b %d"),
            f"{weekly_totals[lightest_week]:.1f} hrs"
        )

        st.subheader("Category Weekly Stats")

        display_stats = category_stats[
            ["avg_per_week", "max_week", "std_dev", "consistency_score"]
        ].sort_values("avg_per_week", ascending=False)

        st.dataframe(display_stats.style.format("{:.2f}"))

        # most intense category-week ever
        max_idx = weekly_pivot.stack().idxmax()
        max_val = weekly_pivot.stack().max()

        st.success(
            f"Most intense week ever: **{max_idx[1]}** — "
            f"{max_val:.1f} hrs (week of {max_idx[0].strftime('%b %d')})"
        )

        # crunch weeks 
        z_scores, crunch = compute_crunch_weeks(weekly_totals)

        if not crunch.empty:
            st.warning("Crunch Weeks Detected")
            for wk, score in crunch.items():
                st.write(
                    f"- Week of {wk.strftime('%b %d')}: "
                    f"{weekly_totals[wk]:.1f} hrs (z={score:.2f})"
                )
        else:
            st.info("No extreme crunch weeks detected")

    # categories
    with tabs[6]:
        cat = df.groupby("Category")["Hours spent"].sum().reset_index()

        fig = px.pie(
            cat,
            names="Category",
            values="Hours spent",
            title="Time Distribution by Category"
        )
        st.plotly_chart(fig, use_container_width=True)

    # holidays
    with tabs[7]:
        h = (
            df.groupby("Is Holiday")["Hours spent"]
            .sum()
            .reset_index()
        )
        h["Is Holiday"] = h["Is Holiday"].map({True: "Holiday", False: "Normal Day"})

        fig = px.bar(
            h,
            x="Is Holiday",
            y="Hours spent",
            title="Holiday vs Normal Day Effort"
        )
        st.plotly_chart(fig, use_container_width=True)

    # raw data
    with tabs[8]:
        st.dataframe(df.sort_values("Date"))

else:
    st.info("Upload your Notion CSV to begin")
