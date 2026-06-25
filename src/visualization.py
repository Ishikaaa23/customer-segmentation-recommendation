"""
Visualization & Dashboard Module
Generates an interactive Plotly HTML dashboard summarising all findings.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#264653", "#A8DADC",
]


def build_executive_dashboard(
    df: pd.DataFrame,
    profile: pd.DataFrame,
    X_pca: np.ndarray,
    segment_recs: dict,
    popularity: pd.DataFrame,
    coverage: pd.DataFrame,
) -> str:
    """
    Build a single self-contained HTML dashboard with:
      - KPI cards
      - Monthly revenue trend
      - Customer segment distribution (pie + scatter)
      - RFM heatmap by segment
      - Top products per segment
      - Recommendation coverage comparison

    Returns the path to the saved HTML file.
    """
    segments = profile["Segment"].unique().tolist()
    seg_colors = {s: SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
                  for i, s in enumerate(sorted(segments))}

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            "Monthly Revenue Trend",
            "Customer Segments in PCA Space",
            "Segment Distribution",
            "Average Spend by Segment",
            "Recency vs Monetary (by Segment)",
            "Frequency Distribution by Segment",
            "Recommendation Coverage: Segment vs Popularity",
            "Top 10 Products – Popularity Baseline",
        ],
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "pie"},     {"type": "bar"}],
            [{"type": "scatter"}, {"type": "box"}],
            [{"type": "bar"},     {"type": "bar"}],
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
    )

    # 1. Monthly Revenue
    monthly = (
        df.groupby("YearMonth")["TotalPrice"].sum().reset_index()
    )
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)
    fig.add_trace(
        go.Scatter(x=monthly["YearMonth"], y=monthly["TotalPrice"],
                   mode="lines+markers", fill="tozeroy",
                   line=dict(color="#4C72B0", width=2),
                   name="Revenue"),
        row=1, col=1,
    )

    # 2. PCA Scatter
    for seg in sorted(profile["Segment"].unique()):
        mask = profile["Segment"] == seg
        fig.add_trace(
            go.Scatter(
                x=X_pca[mask.values, 0],
                y=X_pca[mask.values, 1],
                mode="markers",
                marker=dict(size=4, color=seg_colors[seg], opacity=0.6),
                name=seg,
                legendgroup=seg,
            ),
            row=1, col=2,
        )

    # 3. Pie chart
    counts = profile["Segment"].value_counts()
    fig.add_trace(
        go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            marker_colors=[seg_colors.get(s, "#999") for s in counts.index],
            textinfo="percent+label",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # 4. Average spend by segment
    avg_spend = profile.groupby("Segment")["Monetary"].mean().sort_values(ascending=False)
    fig.add_trace(
        go.Bar(
            x=avg_spend.index.tolist(),
            y=avg_spend.values,
            marker_color=[seg_colors.get(s, "#999") for s in avg_spend.index],
            showlegend=False,
        ),
        row=2, col=2,
    )

    # 5. Recency vs Monetary
    sample = profile.sample(min(2000, len(profile)), random_state=42)
    for seg in sorted(sample["Segment"].unique()):
        s = sample[sample["Segment"] == seg]
        fig.add_trace(
            go.Scatter(
                x=s["Recency"], y=s["Monetary"],
                mode="markers",
                marker=dict(size=5, color=seg_colors[seg], opacity=0.5),
                name=seg, legendgroup=seg, showlegend=False,
            ),
            row=3, col=1,
        )

    # 6. Frequency box per segment
    for seg in sorted(profile["Segment"].unique()):
        fig.add_trace(
            go.Box(
                y=profile.loc[profile["Segment"] == seg, "Frequency"],
                name=seg,
                marker_color=seg_colors[seg],
                showlegend=False,
            ),
            row=3, col=2,
        )

    # 7. Recommendation coverage
    fig.add_trace(
        go.Bar(
            x=coverage["Segment"].tolist(),
            y=coverage["OverlapWithPopularity"].tolist(),
            name="Overlap w/ Popularity",
            marker_color="#B0C4DE",
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=coverage["Segment"].tolist(),
            y=coverage["UniqueToSegment"].tolist(),
            name="Unique to Segment",
            marker_color="#4C72B0",
        ),
        row=4, col=1,
    )

    # 8. Popularity baseline
    pop = popularity.head(10).sort_values("UniqueBuyers", ascending=True)
    short_desc = pop["Description"].str[:30].tolist()
    fig.add_trace(
        go.Bar(
            x=pop["UniqueBuyers"].tolist(),
            y=short_desc,
            orientation="h",
            marker_color="#A8DADC",
            showlegend=False,
        ),
        row=4, col=2,
    )

    # Layout
    fig.update_layout(
        title=dict(
            text="<b>Customer Segmentation & Recommendation Dashboard</b>",
            font=dict(size=22),
            x=0.5,
        ),
        height=1800,
        barmode="stack",
        template="plotly_white",
        legend=dict(orientation="v", x=1.02, y=0.75),
        font=dict(family="Arial", size=11),
    )

    # Axis labels
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_yaxes(title_text="Revenue (£)", row=1, col=1)
    fig.update_xaxes(title_text="PC1", row=1, col=2)
    fig.update_yaxes(title_text="PC2", row=1, col=2)
    fig.update_yaxes(title_text="Avg Spend (£)", row=2, col=2)
    fig.update_xaxes(title_text="Recency (days)", row=3, col=1)
    fig.update_yaxes(title_text="Monetary (£)", row=3, col=1)
    fig.update_xaxes(title_text="Segment", row=3, col=2)
    fig.update_yaxes(title_text="Order Frequency", row=3, col=2)
    fig.update_xaxes(title_text="Unique Buyers", row=4, col=2)

    path = REPORTS_DIR / "dashboard.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"[Dashboard] Interactive dashboard saved → {path}")
    return str(path)


def generate_business_report(
    profile: pd.DataFrame,
    segment_recs: dict,
    coverage: pd.DataFrame,
    best_k: int,
) -> str:
    """
    Write a plain-text markdown business insights report.
    Returns the path to the saved report.
    """
    lines = [
        "# Customer Segmentation — Business Insights Report",
        "",
        f"**Total Customers Analysed:** {len(profile):,}",
        f"**Number of Segments:** {best_k}",
        "",
        "---",
        "",
        "## Segment Overview",
        "",
    ]

    stats_cols = ["Recency", "Frequency", "Monetary", "AvgOrderValue"]
    stats_cols = [c for c in stats_cols if c in profile.columns]
    seg_stats = (
        profile.groupby("Segment")[stats_cols]
        .mean()
        .round(1)
        .reset_index()
    )
    seg_stats["CustomerCount"] = (
        profile.groupby("Segment").size().values
    )
    seg_stats["ShareOfWallet%"] = (
        profile.groupby("Segment")["Monetary"].sum()
        / profile["Monetary"].sum() * 100
    ).round(1).values

    for _, row in seg_stats.iterrows():
        lines += [
            f"### {row['Segment']}",
            f"- **Customers:** {int(row['CustomerCount']):,} "
            f"({row['ShareOfWallet%']:.1f}% of total revenue)",
            f"- Avg Recency: {row.get('Recency', 'N/A')} days | "
            f"Avg Orders: {row.get('Frequency', 'N/A')} | "
            f"Avg Spend: £{row.get('Monetary', 'N/A')}",
            "",
        ]

    lines += [
        "---",
        "",
        "## Marketing Recommendations",
        "",
        "| Segment | Strategy |",
        "|---------|----------|",
        "| High-Value Champions | VIP loyalty programme, early access to new products, "
        "exclusive events |",
        "| Loyal Customers | Personalised thank-you offers, upsell complementary categories |",
        "| Frequent Shoppers | Bundle deals, subscription/auto-replenishment options |",
        "| Bargain Seekers | Flash sales, discount coupons, clearance alerts |",
        "| At-Risk / Hibernating | Win-back campaigns, 'We miss you' discounts (20–30%) |",
        "| New / Promising | Onboarding sequence, first-order incentives, category guides |",
        "| Mid-Tier Customers | Engagement nurturing, loyalty point multipliers |",
        "",
        "---",
        "",
        "## Personalisation Impact",
        "",
        "Segment-based recommendations provide a meaningful personalisation lift "
        "over the popularity baseline:",
        "",
    ]

    for _, row in coverage.iterrows():
        lines.append(
            f"- **{row['Segment']}**: {row['PersonalisationGain%']:.0f}% of recommendations "
            f"are unique to the segment (not in top-10 global popularity)."
        )

    lines += [
        "",
        "---",
        "",
        "## Top Recommended Products by Segment",
        "",
    ]
    for seg, recs in segment_recs.items():
        lines.append(f"### {seg}")
        for i, r in recs.head(5).iterrows():
            lines.append(f"  {i+1}. {r.get('Description', r.get('StockCode', ''))}")
        lines.append("")

    report = "\n".join(lines)
    path = REPORTS_DIR / "business_report.md"
    path.write_text(report)
    print(f"[Report] Business report saved → {path}")
    return str(path)
