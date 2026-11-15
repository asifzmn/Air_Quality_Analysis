"""New figures for the final manuscript (all PLOS-sized via plos_style):
 - boxplot_month           monthly PM2.5 distribution by year (national mean, local time)
 - forecast_performance    RMSE / MdAE vs horizon incl. persistence (values from VERIFIED_NUMBERS.md)
 - validation_embassy      (SI) Berkeley Earth vs US Embassy daily scatter + monthly means
 - pblh_lag                (SI) PBLH-PM2.5 correlation vs lag across zones
 - corr_time_lag           (SI) cross-border lag example (Telangana -> Rangpur)
"""
import sys, ast, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = "/home/asif/Desktop/AQ Resources/PLOS 4th Major Revision/figure_scripts"
sys.path.insert(0, "/home/asif/PycharmProjects/Air_Quality_Analysis"); sys.path.insert(0, HERE)
import plos_style as S
S.setup()
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from data_preparation.spatio_temporal_filtering import read_bd_data_4_years
which = sys.argv[1:] or ["month", "forecast", "valid", "pblh", "lag"]

if "month" in which:
    pm = read_bd_data_4_years()[1]; pm.index = pm.index + pd.Timedelta(hours=6)
    nat = pm.mean(axis=1)["2018":"2021"]
    years = [2018, 2019, 2020, 2021]; cols = ["#cfdbe6", "#9cc0d8", "#5b93c3", "#2c5f8f"]
    fig, ax = plt.subplots(figsize=(S.FULL_W, 3.0))
    w = 0.19; stats = []
    for k, (y, c) in enumerate(zip(years, cols)):
        data = [nat[(nat.index.year == y) & (nat.index.month == m)].dropna().values for m in range(1, 13)]
        pos = np.arange(1, 13) + (k - 1.5) * (w + 0.015)
        ax.boxplot(data, positions=pos, widths=w, whis=(5, 95), showfliers=False, patch_artist=True,
                   medianprops=dict(color=S.EDGE, lw=1.0), whiskerprops=dict(color=S.GREYLINE, lw=0.7),
                   capprops=dict(lw=0), boxprops=dict(facecolor=c, edgecolor=S.EDGE, lw=0.5))
        for m, d in zip(range(1, 13), data):
            stats.append(dict(year=y, month=m, median=np.median(d), q1=np.percentile(d, 25), q3=np.percentile(d, 75)))
    pd.DataFrame(stats).round(2).to_csv(HERE + "/monthly_by_year_stats.csv", index=False)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlim(0.45, 12.55); ax.set_ylabel("Hourly PM$_{2.5}$ ($\\mu$g m$^{-3}$)")
    S.aqi_hbands(ax, 235)
    ax.axhline(65, color=S.EDGE, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.text(7.5, 67, "Bangladesh 24-h standard (65 $\\mu$g m$^{-3}$)", ha="center", va="bottom", fontsize=7, color=S.EDGE)
    ax.legend(handles=[Patch(facecolor=c, edgecolor=S.EDGE, lw=0.5, label=str(y)) for y, c in zip(years, cols)],
              frameon=False, loc="upper center", ncol=4)
    S.save(fig, "boxplot_month")

if "forecast" in which:
    rmse = {"Persistence": [0.077, 0.106, 0.111, 0.130, 0.139, 0.147, 0.144],
            "Bayesian Ridge": [0.077, 0.105, 0.114, 0.120, 0.123, 0.125, 0.125],
            "LightGBM": [0.086, 0.119, 0.125, 0.126, 0.126, 0.127, 0.126],
            "MLP": [0.092, 0.117, 0.120, 0.122, 0.126, 0.125, 0.133],
            "CNN": [0.113, 0.121, 0.124, 0.128, 0.130, 0.131, 0.135],
            "LSTM encoder-decoder": [0.119, 0.122, 0.123, 0.123, 0.126, 0.125, 0.128],
            "ConvLSTM": [0.088, 0.113, 0.122, 0.132, 0.137, 0.147, 0.150]}
    # MdAE (normalised); MAPE dropped: the notebook computed MAPE(real + 0.1, pred), i.e. offset on the truth only -> invalid
    mdae = {"Persistence": [0.039, 0.045, 0.057, 0.066, 0.066, 0.069, 0.078],
            "Bayesian Ridge": [0.042, 0.052, 0.060, 0.065, 0.071, 0.072, 0.077],
            "LightGBM": [0.042, 0.055, 0.062, 0.067, 0.063, 0.066, 0.061],
            "MLP": [0.059, 0.072, 0.067, 0.064, 0.073, 0.072, 0.077],
            "CNN": [0.058, 0.060, 0.054, 0.061, 0.064, 0.068, 0.068],
            "LSTM encoder-decoder": [0.060, 0.068, 0.066, 0.067, 0.065, 0.071, 0.070],
            "ConvLSTM": [0.054, 0.061, 0.069, 0.078, 0.073, 0.086, 0.080]}
    # (color, linestyle, marker, linewidth, marker face): blues only, no red; models separated by dash pattern + marker
    sty = {"Persistence": ("#111111", "-", "s", 2.4, "#111111"),   # thick solid black: the baseline (finalizer P4)
           "Bayesian Ridge": (S.DEEP, "-", "s", 1.9, S.DEEP),
           "LightGBM": ("#5b93c3", (0, (6, 1.5, 1.5, 1.5)), "^", 1.2, "white"),
           "MLP": (S.GREYLINE, (0, (1, 1.3)), "v", 1.3, S.GREYLINE),
           "CNN": (S.LIGHT, (0, (3, 1.2)), "D", 1.3, "white"),
           "LSTM encoder-decoder": (S.EDGE, "-", "o", 1.4, "white"),
           "ConvLSTM": (S.SLATE, (0, (5, 1.2, 1, 1.2, 1, 1.2)), "P", 1.2, S.SLATE)}
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL_W, 2.9))
    d = np.arange(1, 8)
    for ax, (vals, yl, t) in zip(axes, [(rmse, "RMSE (normalized)", "(a) RMSE"), (mdae, "MdAE (normalized)", "(b) MdAE")]):
        for name, v in vals.items():
            c, l, mk, lw, mf = sty[name]
            ax.plot(d, v, color=c, ls=l, marker=mk, ms=4.0, lw=lw, mfc=mf, mec=c, mew=0.9, label=name,
                    zorder=4 if name in ("Persistence", "Bayesian Ridge") else 3)
        ax.set_xticks(d); ax.set_xlabel("Forecast horizon (days)"); ax.set_ylabel(yl); ax.set_title(t, loc="left"); S.ygrid(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    axes[0].legend(h, l, loc="lower left", ncol=4, frameon=False, bbox_to_anchor=(0.0, 1.10), borderaxespad=0, handlelength=3.2)
    S.save(fig, "forecast_performance")

if "valid" in which:
    j = pd.read_csv(HERE + "/../validation_us_embassy/daily_paired_dhaka.csv", index_col=0, parse_dates=True)
    e, b = j["US Embassy"], j["Berkeley Earth"]
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL_W, 3.0), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]; lim = 400
    ax.scatter(e, b, s=5, color=S.LIGHT, edgecolor=S.EDGE, linewidth=0.2, alpha=0.8)
    ax.plot([0, lim], [0, lim], color="#888", lw=0.8, ls=(0, (4, 2)), label="1:1")
    sl, ic = np.polyfit(e, b, 1); ax.plot([0, lim], [ic, ic + sl * lim], color=S.DEEP, lw=1.3, label=f"OLS: y = {sl:.2f}x + {ic:.1f}")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("US Embassy daily PM$_{2.5}$ ($\\mu$g m$^{-3}$)"); ax.set_ylabel("Berkeley Earth daily PM$_{2.5}$ ($\\mu$g m$^{-3}$)")
    ax.set_title(f"(a) Daily means, n = {len(j):,}, r = {e.corr(b):.2f}", loc="left"); ax.legend(frameon=False, loc="upper left")
    ax = axes[1]; mm = j.resample("M").mean()
    ax.plot(mm.index, mm["US Embassy"], color=S.EDGE, lw=1.4, label="US Embassy (reference monitor)")
    ax.plot(mm.index, mm["Berkeley Earth"], color=S.STEEL, lw=1.4, ls=(0, (4, 2)), label="Berkeley Earth (Dhaka zone)")
    ax.set_ylabel("Monthly mean PM$_{2.5}$ ($\\mu$g m$^{-3}$)"); ax.set_title("(b) Monthly means, 2018–2021", loc="left")
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); ax.legend(frameon=False, loc="upper center")
    S.ygrid(ax); fig.tight_layout(); S.save(fig, "validation_embassy")

if "pblh" in which:
    r = pd.read_csv(HERE + "/pblh_lag_corr.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(S.COL_W, 2.8))
    q = r.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
    ax.fill_between(q.index, q[0.05], q[0.95], color=S.GREYBLUE, alpha=0.6, lw=0, label="5th–95th percentile of zones")
    ax.fill_between(q.index, q[0.25], q[0.75], color=S.LIGHT, alpha=0.9, lw=0, label="Interquartile range")
    ax.plot(q.index, q[0.5], color=S.DEEP, lw=1.6, label="Median of 55 zones")
    ax.axvline(3, color=S.EDGE, lw=0.8, ls=(0, (3, 2))); ax.text(3.4, -0.02, "3 h", fontsize=7.5, color=S.EDGE, va="top")
    ax.set_xlabel("Lag of PBLH (hours)"); ax.set_ylabel("Pearson r (PBLH vs PM$_{2.5}$)"); ax.set_xticks(range(0, 25, 3)); ax.set_xlim(0, 24)
    ax.axhline(0, color="#999", lw=0.6); S.ygrid(ax); ax.legend(frameon=False, loc="lower right")
    S.save(fig, "PBLH_analysis")

if "lag" in which:
    from data_preparation import read_all_granularity_data
    reg = read_all_granularity_data()[3]
    ev = pd.read_csv("/home/asif/PycharmProjects/Air_Quality_Analysis/cross_correlation/evidence_df.csv", parse_dates=["start_date", "end_date"])
    e = ev.loc[82246]; L = int(e.max_correlation_lag); bars = ast.literal_eval(e.all_lags_correlation)
    sub = lambda s: s.loc[e.start_date:e.end_date]
    fw, lw_, ls_ = sub(reg[e.bangladesh_zone]), sub(reg[e.india_zone]), sub(reg[e.india_zone].shift(L))
    off = pd.Timedelta(hours=6)          # plot in local time
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.FULL_W, 2.7), gridspec_kw={"width_ratios": [1, 3.4]})
    a1.bar(range(7), bars, color=[S.DEEP if k == L else "#d0d0d0" for k in range(7)], edgecolor=S.EDGE, lw=0.5)
    for k in (0, L): a1.text(k, bars[k] + 0.012, f"{bars[k]:.2f}", ha="center", va="bottom", fontsize=7)
    a1.set_xticks(range(7)); a1.set_xlabel("Lag (hours)"); a1.set_ylabel("Pearson r"); a1.set_ylim(0, max(bars) * 1.18); a1.set_title("(a) Correlation by lag", loc="left")
    a2.plot(fw.index + off, fw.values, color="#7a7a7a", lw=1.2, label=f"{e.bangladesh_zone} (follower)")
    a2.plot(lw_.index + off, lw_.values, color=S.DEEP, lw=1.2, label=f"{e.india_zone} (leader)")
    a2.plot(ls_.index + off, ls_.values, color=S.DEEP, lw=1.2, ls=(0, (4, 2)), label=f"{e.india_zone} shifted by {L} h")
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); a2.set_ylabel("PM$_{2.5}$ ($\\mu$g m$^{-3}$)")
    a2.set_title(f"(b) Hourly PM$_{{2.5}}$, {e.start_date + off:%d}–{e.end_date + off:%d %b %Y} (local time)", loc="left")
    for ax in (a1, a2): S.ygrid(ax)
    fig.tight_layout(); a2.legend(frameon=False, loc="lower left", ncol=3, bbox_to_anchor=(0, 1.12), borderaxespad=0)
    S.save(fig, "corr_time_lag")
