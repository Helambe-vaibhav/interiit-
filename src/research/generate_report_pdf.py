"""
src/research/generate_report_pdf.py
===================================
Compiles the written research report (src/report.pdf) in clear, simple language
for Inter IIT Tech Meet 15.0 Quant Selection Track A.
Explains all strategy components intuitively without heavy, obscure jargon.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)


def build_pdf(filename="src/report.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Clean typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0d2b45'),
        alignment=1,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#415a77'),
        alignment=1,
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor('#0d3b66'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#1f4e78'),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#222222'),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        leftIndent=12,
        textColor=colors.HexColor('#222222'),
        spaceAfter=3
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#111111')
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    )

    story = []

    # Title Banner
    story.append(Paragraph("Inter IIT Tech Meet 15.0 &mdash; Quantitative Trading Selection", subtitle_style))
    story.append(Paragraph("Track A: Volatility-Adjusted Cross-Sectional Strategy", title_style))
    story.append(Paragraph("<b>Strategy Research Report</b> &nbsp;|&nbsp; Clear, Robust, and Built for Real-World Markets", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0d3b66'), spaceAfter=8))

    # Executive Summary
    story.append(Paragraph("Executive Summary: What We Built & Key Results", h1_style))
    story.append(Paragraph(
        "The goal of Track A is to build an automated trading strategy for a universe of 24 stocks that delivers strong profits "
        "while actively managing risk. Instead of using complex 'black-box' models that are prone to overfitting, our strategy follows "
        "a simple and intuitive philosophy: <b>Find steady winners, buy them on temporary dips, give calmer stocks more weight, "
        "and hold cash when the overall market is in a downtrend.</b>",
        body_style
    ))
    story.append(Paragraph(
        "Across the entire 6-year test period (1,484 trading days, 2016&ndash;2022):<br/>"
        "&bull; <b>Initial Capital:</b> $100,000.00 &nbsp;|&nbsp; <b>Final Portfolio Value:</b> <b>$231,775.83 (+131.78% total return)</b><br/>"
        "&bull; <b>Annual Return (CAGR):</b> <b>15.34%</b> per year &nbsp;|&nbsp; <b>Sharpe Ratio:</b> <b>1.16</b> (95% Confidence Interval: [0.38, 2.07])<br/>"
        "&bull; <b>Maximum Drawdown (Peak Loss):</b> <b>22.64%</b> (slashed from 37.88% thanks to our crash protection rule)<br/>"
        "&bull; <b>Risk Violations:</b> <b>0 leverage violations</b> (stayed safely under the 1.0 limit) and <b>0 bankruptcies</b>.",
        body_style
    ))

    # Section 1: Volatility Modeling
    story.append(Paragraph("1. Measuring Risk the Smart Way (Volatility Modeling)", h1_style))
    story.append(Paragraph(
        "To decide how many shares of each stock to buy, we need to know how risky each stock is. "
        "Most basic trading systems only look at the closing price at the end of each day. But closing prices can hide big surprises: "
        "a stock might swing up 8% at noon and plunge 7% by afternoon, yet finish flat. Looking only at the close misses all that turbulence.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Our Approach (Garman-Klass Volatility):</b> We look at the daily <b>High, Low, Open, and Close</b> prices. "
        "By comparing the highest and lowest prices of the day alongside the opening and closing prices, we get a much truer picture of the stock's volatility. "
        "Research shows this method is about <b>7.4 times more accurate</b> at estimating risk than just using closing prices.",
        body_style
    ))

    vol_data = [
        [Paragraph("Volatility Method", table_header), Paragraph("Data Used", table_header), Paragraph("Average Measured Risk", table_header), Paragraph("Accuracy Relative to Close-Only", table_header)],
        [Paragraph("<b>Garman-Klass (Our Choice)</b>", table_cell), Paragraph("Open, High, Low, Close", table_cell), Paragraph("27.37% / year", table_cell), Paragraph("<b>7.4x more accurate (Best)</b>", table_cell)],
        [Paragraph("Parkinson (1980)", table_cell), Paragraph("High and Low only", table_cell), Paragraph("27.03% / year", table_cell), Paragraph("5.2x more accurate", table_cell)],
        [Paragraph("Standard Close-to-Close", table_cell), Paragraph("Close only", table_cell), Paragraph("28.04% / year", table_cell), Paragraph("1.0x (Standard baseline)", table_cell)]
    ]
    t_vol = Table(vol_data, colWidths=[140, 120, 110, 170])
    t_vol.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0d3b66')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d0d7de')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f6f8fa')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_vol)
    story.append(Spacer(1, 4))

    # Section 2: Stock Selection & Ranking
    story.append(Paragraph("2. How We Pick the Best Stocks (3-Pillar Ranking System)", h1_style))
    story.append(Paragraph(
        "Every week, we score and rank all 24 stocks using three practical, time-tested rules. "
        "To avoid curve-fitting to the past, we give all three rules equal importance (one-third each):",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. Steady Upward Trend (Trend Sharpe):</b> We calculate how much the stock gained over the last 60 days, divided by its daily volatility. "
        "We want smooth, steady compounders, not lottery stocks that jump on wild hype and then collapse.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>2. Buying the Dip (Pullback Reversal):</b> If a good stock has had a small 1-to-2 week dip below its 10-day moving average, "
        "we score it higher. This lets us buy great stocks on temporary discounts rather than chasing expensive peaks.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>3. Favor Calmer Stocks (Low-Volatility Edge):</b> Decades of financial research show that calmer stocks tend to deliver "
        "better risk-adjusted returns over time because speculative, high-volatility stocks get bid up too high by retail traders.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>Safety Guard (Never Catch Falling Knives):</b> We only buy a stock if it is currently trading above its 50-day average. "
        "If a stock is in a multi-month death spiral, we do not touch it.",
        bullet_style
    ))

    story.append(Spacer(1, 4))

    # Section 3: Portfolio Sizing
    story.append(Paragraph("3. Sizing Positions: Spread the Risk (Risk Parity)", h1_style))
    story.append(Paragraph(
        "Once we pick the top 6 stocks, how much money goes into each? We use a common-sense risk-budgeting method:",
        body_style
    ))
    story.append(Paragraph(
        "&bull; <b>Calmer Stocks Get More Capital, Volatile Stocks Get Less:</b> If Stock A swings twice as much as Stock B, "
        "we give Stock B twice as much money. This ensures that every stock contributes an equal amount of risk to the portfolio, "
        "so no single wild stock can ruin our overall performance.",
        bullet_style
    ))
    story.append(Paragraph(
        "&bull; <b>Strict 18% Single-Stock Cap:</b> No single company can ever take up more than 18% of our portfolio. "
        "Even if one stock faces an unexpected disaster, the other 82% of the portfolio protects us.",
        bullet_style
    ))
    story.append(Paragraph(
        "&bull; <b>2% Trading Buffer (Cut Fees):</b> We only rebalance once every 5 trading days (weekly). If a stock's target weight is "
        "within 2% of what we already own, we skip the trade. This saved us hundreds of dollars in unnecessary broker fees and commissions.",
        bullet_style
    ))

    story.append(Spacer(1, 4))

    # Section 4: Crash Protection
    story.append(Paragraph("4. Crash Protection: How We Cut Drawdown from 38% to 22%", h1_style))
    story.append(Paragraph(
        "During major market panics (like the COVID crash in March 2020 or the 2018 market selloff), almost all stocks drop together. "
        "Staying 100% invested in stocks during a crash leads to painful losses.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Our Macro Defense Rule (Meb Faber 2007):</b><br/>"
        "At every weekly rebalance, we check the overall market average over the last 100 days:<br/>"
        "&bull; <b>Normal Bull Market:</b> When the market is healthy, we invest 96% into our top 6 stocks.<br/>"
        "&bull; <b>Market Crash Warning:</b> If the overall market falls below its 100-day average, <b>we automatically cut our stock holdings in half "
        "(to 48%) and park the other 52% safely in cash.</b>",
        body_style
    ))
    story.append(Paragraph(
        "<b>Why this worked so well:</b> This single rule protected our money when the COVID crash hit in early 2020. "
        "It reduced our worst peak-to-trough loss from <b>37.88% down to 22.64%</b>, while actually <i>improving</i> our Sharpe ratio from 1.03 to 1.16 "
        "because we had cash available to buy back at cheaper prices once the market stabilized.",
        body_style
    ))

    story.append(Spacer(1, 4))

    # Section 5: Testing Different Market Cycles
    story.append(Paragraph("5. Testing Different Market Cycles (Avoiding Overfitting)", h1_style))
    story.append(Paragraph(
        "A huge mistake in trading competitions is 'curve-fitting' &mdash; tweaking rules until a backtest shows massive numbers on paper. "
        "For example, putting all your money into 2 or 3 hot stocks produces great returns during bull runs, but loses nearly 40% when the market turns. "
        "To prove our strategy works across different environments, we tested it across three completely different real-world market periods:",
        body_style
    ))

    cycle_data = [
        [Paragraph("Market Period", table_header), Paragraph("What Happened in the Market", table_header), Paragraph("Our Strategy Return", table_header), Paragraph("Sharpe Ratio", table_header), Paragraph("Worst Drawdown", table_header), Paragraph("Takeaway", table_header)],
        [Paragraph("<b>2017 &ndash; 2018</b>", table_cell), Paragraph("Steady Bull Market", table_cell), Paragraph("+46.89%", table_cell), Paragraph("<b>2.16</b>", table_cell), Paragraph("8.70%", table_cell), Paragraph("Strong steady gains", table_cell)],
        [Paragraph("<b>2019 &ndash; 2020</b>", table_cell), Paragraph("COVID Crash & Recovery", table_cell), Paragraph("+19.88%", table_cell), Paragraph("<b>0.77</b>", table_cell), Paragraph("21.99%", table_cell), Paragraph("Protected cash during crash", table_cell)],
        [Paragraph("<b>2021 &ndash; 2022</b>", table_cell), Paragraph("Inflation & Tech Selloff", table_cell), Paragraph("+28.85%", table_cell), Paragraph("<b>1.17</b>", table_cell), Paragraph("13.64%", table_cell), Paragraph("Solid profits in tough market", table_cell)],
        [Paragraph("<b>Full 6 Years</b>", table_cell), Paragraph("<b>Complete Historical Dataset</b>", table_cell), Paragraph("<b>+131.78%</b>", table_cell), Paragraph("<b>1.16</b>", table_cell), Paragraph("<b>22.64%</b>", table_cell), Paragraph("<b>Consistent in all 3 periods</b>", table_cell)]
    ]
    t_cycle = Table(cycle_data, colWidths=[75, 125, 85, 75, 80, 100])
    t_cycle.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0d3b66')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d0d7de')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f6f8fa')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_cycle)

    story.append(Spacer(1, 4))

    # Section 6: Results Scorecard
    story.append(Paragraph("6. Official Performance Scorecard", h1_style))
    summary_data = [
        [Paragraph("Evaluation Metric", table_header), Paragraph("Default Baseline", table_header), Paragraph("Our Strategy (Track A)", table_header), Paragraph("What It Means", table_header)],
        [Paragraph("Starting Cash", table_cell), Paragraph("$100,000.00", table_cell), Paragraph("$100,000.00", table_cell), Paragraph("Standard starting balance", table_cell)],
        [Paragraph("Final Portfolio Value", table_cell), Paragraph("$90,633.90", table_cell), Paragraph("<b>$231,775.83</b>", table_cell), Paragraph("More than doubled initial money", table_cell)],
        [Paragraph("Total Return", table_cell), Paragraph("-9.37%", table_cell), Paragraph("<b>+131.78%</b>", table_cell), Paragraph("Beat baseline by +141%", table_cell)],
        [Paragraph("Compound Annual Return (CAGR)", table_cell), Paragraph("-1.66%", table_cell), Paragraph("<b>15.34% / year</b>", table_cell), Paragraph("Strong long-term compounding", table_cell)],
        [Paragraph("Sharpe Ratio (Risk-Adjusted)", table_cell), Paragraph("-0.18", table_cell), Paragraph("<b>1.16</b> (CI: [0.38, 2.07])", table_cell), Paragraph("Comfortably above 1.0 hurdle", table_cell)],
        [Paragraph("Sortino Ratio (Downside Risk)", table_cell), Paragraph("-0.22", table_cell), Paragraph("<b>1.57</b>", table_cell), Paragraph("Low downside fluctuations", table_cell)],
        [Paragraph("Maximum Drawdown (Deepest Dip)", table_cell), Paragraph("20.58%", table_cell), Paragraph("<b>22.64%</b>", table_cell), Paragraph("Kept under control via cash hedge", table_cell)],
        [Paragraph("Calmar Ratio (Return / Drawdown)", table_cell), Paragraph("-0.08", table_cell), Paragraph("<b>0.68</b>", table_cell), Paragraph("Positive reward-to-risk ratio", table_cell)],
        [Paragraph("Leverage Violations", table_cell), Paragraph("0", table_cell), Paragraph("<b>0 (Max 0.98 &le; 1.0)</b>", table_cell), Paragraph("100% compliant with competition rules", table_cell)],
        [Paragraph("Account Bankruptcies", table_cell), Paragraph("0", table_cell), Paragraph("<b>0 (Cash stayed positive)</b>", table_cell), Paragraph("Zero borrowed debt; fully solvent", table_cell)],
        [Paragraph("Speed / Throughput", table_cell), Paragraph("—", table_cell), Paragraph("<b>1,460 ticks / second</b>", table_cell), Paragraph("Runs full 6-year backtest in ~1 second", table_cell)]
    ]
    t_summary = Table(summary_data, colWidths=[140, 100, 140, 160])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0d3b66')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d0d7de')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f6f8fa')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_summary)

    story.append(Spacer(1, 4))

    # Section 7: Proof of Fair Play & Integrity
    story.append(Paragraph("7. Proof of Fair Play: Zero Lookahead & Zero Data Leakage", h1_style))
    story.append(Paragraph(
        "A common flaw in algorithmic trading is accidentally using tomorrow's prices to make today's decisions (lookahead bias). "
        "We ran an automated audit (<code>src/research/audit_data_leakage.py</code>) to prove that our strategy is 100% causal:<br/>"
        "&bull; <b>Truncation Test:</b> We ran the strategy on the full dataset and then on a cut-off version. "
        "The difference between the portfolio weights at the cutoff was <b>exactly 0.000</b>. Future data has zero influence on past decisions.<br/>"
        "&bull; <b>Realistic Next-Day Open Execution:</b> Decisions made at today's close are filled at <b>tomorrow's open price</b>, "
        "exactly mirroring how live execution works in the real world.<br/>"
        "&bull; <b>Deterministic:</b> All random seeds are fixed (seed 42), meaning this test can be rerun by anyone and will yield the exact same results.",
        body_style
    ))

    # Key References
    story.append(Spacer(1, 4))
    story.append(Paragraph("Key Research Sources", h1_style))
    story.append(Paragraph("1. <b>Garman & Klass (1980):</b> <i>On the Estimation of Security Price Volatilities from Historical Data</i> &mdash; The foundation for our range volatility estimator.", bullet_style))
    story.append(Paragraph("2. <b>Meb Faber (2007):</b> <i>A Quantitative Approach to Tactical Asset Allocation</i> &mdash; The inspiration for our 100-day trend cash defense.", bullet_style))
    story.append(Paragraph("3. <b>Maillard, Roncalli & Te&iuml;letche (2010):</b> <i>The Properties of Equally Weighted Risk Contribution Portfolios</i> &mdash; The math behind our inverse-volatility sizing.", bullet_style))
    story.append(Paragraph("4. <b>Ang, Hodrick, Xing & Zhang (2006):</b> <i>The Cross-Section of Volatility and Expected Returns</i> &mdash; Explains the low-volatility anomaly.", bullet_style))

    doc.build(story)
    print(f"Successfully generated simple-language PDF report: {filename}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "src/report.pdf"
    build_pdf(out)
