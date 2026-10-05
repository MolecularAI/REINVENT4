import marimo

__generated_with = "0.23.6"
app = marimo.App(
    width="full",
    app_title="Sampled Plots",
    auto_download=["html"],
)


@app.cell
def _():
    import argparse
    import pathlib

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    # Configure Altair for better interactivity
    alt.data_transformers.disable_max_rows()

    PLOT_WIDTH = 600
    PLOT_HEIGHT = 400
    SCATTER_WIDTH = 840
    SCATTER_HEIGHT = 620
    FACET_PLOT_WIDTH = 340
    FACET_PLOT_HEIGHT = 240
    FACET_COLUMNS = 3
    METHOD_LEGEND = alt.Legend(
        title=None,
        orient="bottom",
        direction="horizontal",
        columns=4,
        titleOrient="left",
        labelFontSize=12,
        titleFontSize=15,
        symbolSize=160,
        labelLimit=240,
    )


    def method_color(*, sort=None):
        return alt.Color(
            "method:N",
            sort=sort,
            legend=METHOD_LEGEND,
        )


    def run_color(*, sort=None):
        return alt.Color("Run:N", title="Run", sort=sort, legend=None)

    return (
        SCATTER_HEIGHT,
        SCATTER_WIDTH,
        alt,
        argparse,
        method_color,
        mo,
        np,
        pathlib,
        pl,
    )


@app.cell
def _(argparse, pathlib):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./runs")
    args, _ = parser.parse_known_args()

    runtime_initial_path = pathlib.Path(args.data_dir)
    return (runtime_initial_path,)


@app.cell
def _(mo, runtime_initial_path):
    run_button_file = mo.ui.run_button(label="Continue")

    file_browser = mo.ui.file_browser(
        initial_path=runtime_initial_path,
        selection_mode="directory",
        label="Select Run Directory",
    )
    mo.vstack([file_browser, run_button_file])
    return file_browser, run_button_file


@app.cell
def _(file_browser, mo, run_button_file, runtime_initial_path):
    mo.stop(not run_button_file.value)
    root_path = (
        file_browser.value[0].path
        if len(file_browser.value)
        else runtime_initial_path
    )
    root_name = root_path.name
    print("Selected", root_path, root_name)
    return root_name, root_path


@app.cell
def _(pathlib, root_path):
    run_dirs_all: dict[str, tuple[pathlib.Path, list[pathlib.Path]]] = {}

    for level1_dir in root_path.iterdir():
        if level1_dir.is_dir():
            _dir = [item for item in level1_dir.iterdir() if item.is_dir()]
            _run_id = str(level1_dir.relative_to(root_path))
            run_dirs_all[_run_id] = level1_dir, sorted(_dir)
    return (run_dirs_all,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Completeness Check

    Before running the analysis, each run directory is checked for the required
    files. The accordion below shows per-method status with indicator:

    - **sampled** — does every run have `sampled_hits.csv` (binned sample for diversity metrics)?
    - **cum_div** — does every run have `cumulative_diversity.csv` (diversity over time)?

    | Icon | Meaning |
    |------|---------|
    | ✅ | All runs pass |
    | 🟠 | Some runs pass, some missing |
    | ❌ | No runs pass |
    """)
    return


@app.cell
def _(
    mo,
    pathlib,
    pl,
    run_dirs_all: "dict[str, tuple[pathlib.Path, list[pathlib.Path]]]",
):
    # Data completeness check
    def _nrows(p: pathlib.Path) -> int:
        if not p.exists():
            return 0
        n = pl.scan_csv(p).select(pl.len()).collect().item()
        return int(n)


    check_rows = []
    for _method, (_, _dirs) in run_dirs_all.items():
        for _rd in _dirs:
            check_rows.append(
                {
                    "method": _method,
                    "run": _rd.name,
                    "sampled_hits": _nrows(_rd / "sampled_hits.csv"),
                    "cum_diversity": _nrows(_rd / "cumulative_diversity.csv"),
                }
            )

    check_df = pl.DataFrame(check_rows)


    def _icon(passed: int, total: int) -> str:
        if passed == total:
            return "✅"
        if passed > 0:
            return "🟠"
        return "❌"


    details = {}
    for m in sorted(check_df["method"].unique().to_list()):
        mdf = check_df.filter(pl.col("method") == m)
        n = len(mdf)
        label = (
            f"{m} ({n} runs) "
            f"| sampled:{_icon((mdf['sampled_hits'] > 0).sum(), n)} "
            f"cum_div:{_icon((mdf['cum_diversity'] > 0).sum(), n)}"
        )
        details[label] = mo.ui.table(mdf.to_dicts(), selection=None)

    mo.vstack(
        [
            mo.md(f"### Data Check — ({len(check_df)} runs)"),
            mo.accordion(details),
        ]
    )
    return


@app.cell
def _(mo, run_dirs_all: "dict[str, tuple[pathlib.Path, list[pathlib.Path]]]"):
    all_run_keys = sorted(run_dirs_all.keys())

    run_keys_multiselect = mo.ui.multiselect(
        options=all_run_keys,
        value=all_run_keys,
        label="Select Methods to include",
    )

    run_button_check = mo.ui.run_button(label="Continue")
    mo.vstack([run_keys_multiselect, run_button_check])
    return all_run_keys, run_button_check, run_keys_multiselect


@app.cell
def _(
    all_run_keys,
    mo,
    run_button_check,
    run_dirs_all: "dict[str, tuple[pathlib.Path, list[pathlib.Path]]]",
    run_keys_multiselect,
):
    mo.stop(not run_button_check.value)
    selected_run_keys: list[str] = run_keys_multiselect.value or all_run_keys
    run_dirs = {k: run_dirs_all[k] for k in selected_run_keys}
    mo.md(
        f"Using **{len(selected_run_keys)} / {len(all_run_keys)}** methods: {', '.join(selected_run_keys)}"
    )
    return (run_dirs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Chemical Space Visualisation

    Projects downsampled hit molecules into 2D using Morgan fingerprints (radius 2, 2048 bits)
    and two complementary dimensionality reduction methods:

    - **PCA**: linear projection onto the two principal components of variance.
      Useful for seeing global spread and overlap between methods.
    - **t-SNE**: non-linear embedding that preserves local neighbourhood structure.
      Clusters indicate groups of structurally similar molecules.

    This uses the 'sampled_hits.csv' produced by the script 'preprocess_single.py'.
    """)
    return


@app.cell
def _(pl, run_dirs):
    _all_runs = []
    for _method, (_, _dirs) in run_dirs.items():
        for _rd in _dirs:
            csv_path = _rd / "sampled_hits.csv"
            if csv_path.exists():
                try:
                    # Ensure the run file contains SMILES
                    df = pl.read_csv(csv_path)
                    if "SMILES" in df.columns:
                        df = df.with_columns(
                            pl.lit(_method).alias("method"),
                            pl.lit(_rd.name).alias("run_id"),
                        )
                        _all_runs.append(df)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

    if _all_runs:
        sampled_hits_df = pl.concat(_all_runs, how="diagonal_relaxed")
    else:
        sampled_hits_df = pl.DataFrame()

    sampled_hits_df
    return (sampled_hits_df,)


@app.cell
def _(mo, sampled_hits_df):
    chem_methods = (
        sampled_hits_df["method"].unique().sort().to_list()
        if not sampled_hits_df.is_empty()
        else []
    )
    chem_method_select = mo.ui.multiselect(
        options=chem_methods,
        value=chem_methods,
        label="Select methods for chemical space analysis",
    )
    samples_per_run_bin = mo.ui.number(
        label="Samples per run and bin",
        value=20,
        step=1,
    )
    mo.md(
        "Sets the maximum number of molecules to keep **per run and step_bin** for "
        "the chemical-space plots. This controls the downsampling so large runs don't "
        "dominate the visualization."
    )
    chem_run_button = mo.ui.run_button(label="Continue")
    mo.vstack([chem_method_select, samples_per_run_bin, chem_run_button])
    return (
        chem_method_select,
        chem_methods,
        chem_run_button,
        samples_per_run_bin,
    )


@app.cell
def _(
    chem_method_select,
    chem_methods,
    chem_run_button,
    mo,
    pl,
    sampled_hits_df,
    samples_per_run_bin,
):
    mo.stop(not chem_run_button.value)

    from rdkit import Chem
    from rdkit.Chem import AllChem
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE


    def smiles_to_fp(smi, radius=2, nbits=2048):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return list(
            AllChem.GetMorganGenerator(radius=radius, fpSize=nbits).GetFingerprint(
                mol
            )
        )


    _selected_methods = chem_method_select.value or chem_methods

    max_samples = max(1, int(samples_per_run_bin.value or 20))

    # Downsample again to max N per run, method, and step_bin to keep it manageable
    downsampled_df = (
        sampled_hits_df.filter(pl.col("method").is_in(_selected_methods))
        .group_by(["run_id", "method", "step_bin"])
        .head(max_samples)
    )

    print("Sampled for analysis:", downsampled_df.shape)
    downsampled_df.group_by(["step_bin", "method"]).count()
    return PCA, TSNE, downsampled_df, smiles_to_fp


@app.cell
def _(downsampled_df, np, pl, smiles_to_fp):
    # Compute fingerprints, filter failed parses
    fps_df = downsampled_df.with_columns(
        pl.col("SMILES")
        .map_elements(smiles_to_fp, return_dtype=pl.List(pl.Int8))
        .alias("fp")
    ).filter(pl.col("fp").is_not_null())

    fps_X = np.array(fps_df["fp"].to_list())

    print("Fingerprints ready:", fps_X.shape)
    return fps_X, fps_df


@app.cell
def _(PCA, fps_X, fps_df, pl):
    _pca = PCA(n_components=2, random_state=42)
    pca_emb = _pca.fit_transform(fps_X)

    pca_df = fps_df.drop("fp").with_columns(
        [
            pl.Series("pca_x", pca_emb[:, 0]),
            pl.Series("pca_y", pca_emb[:, 1]),
        ]
    )
    return pca_df, pca_emb


@app.cell
def _(SCATTER_HEIGHT, SCATTER_WIDTH, alt, method_color):
    def build_scatter_plot(data, x_col, y_col, title, step_limit):
        safe_limit = max(1, int(step_limit))
        axis_titles = {
            "pca_x": "PC1",
            "pca_y": "PC2",
            "tsne_x": "t-SNE 1",
            "tsne_y": "t-SNE 2",
        }
        step_param = alt.param(
            "step_limit",
            value=safe_limit,
            bind=alt.binding_range(min=1, max=safe_limit, step=1),
        )

        return (
            alt.Chart(data.sort("step"))
            .add_params(step_param)
            .transform_filter("datum.step <= step_limit")
            .transform_calculate(
                opacity=(
                    "max(0.1, min(0.8, 0.1 + 0.7*(datum.step/step_limit)))"
                )
            )
            .mark_circle(size=40)
            .encode(
                x=alt.X(f"{x_col}:Q", title=axis_titles.get(x_col, x_col)),
                y=alt.Y(f"{y_col}:Q", title=axis_titles.get(y_col, y_col)),
                color=method_color(),
                opacity=alt.Opacity("opacity:Q", legend=None),
                tooltip=[
                    "SMILES:N",
                    "method:N",
                    "run_id:N",
                    "Score:Q",
                    "step_bin:Q",
                    "step",
                ],
            )
            .properties(
                width=SCATTER_WIDTH,
                height=SCATTER_HEIGHT,
                title=title,
            )
        )

    return (build_scatter_plot,)


@app.cell
def _(build_scatter_plot, mo, pca_df, root_name):
    _step_limit = int(pca_df["step"].max()) if not pca_df.is_empty() else 1

    # Plot PCA
    pca_chart = build_scatter_plot(
        pca_df,
        "pca_x",
        "pca_y",
        f"PCA of Hit Fingerprints for {root_name}",
        _step_limit,
    )

    save_pca_button = mo.ui.run_button(label="Save PCA plots")

    mo.vstack([mo.ui.altair_chart(pca_chart), save_pca_button])
    return pca_chart, save_pca_button


@app.cell
def _(mo, pca_chart, root_path, save_pca_button):
    mo.stop(not save_pca_button.value)
    pca_chart.save(root_path / "pca_diverse_hits.html")
    pca_chart.save(root_path / "pca_diverse_hits.svg")
    pca_chart.save(root_path / "pca_diverse_hits.png", scale_factor=2.0)
    return


@app.cell
def _(TSNE, fps_X, fps_df, mo, pca_emb, pl, save_pca_button):
    mo.stop(save_pca_button.value)
    _tsne = TSNE(n_components=2, random_state=42, n_jobs=-1, init=pca_emb)
    _tsne_emb = _tsne.fit_transform(fps_X)

    tsne_df = fps_df.drop("fp").with_columns(
        [
            pl.Series("tsne_x", _tsne_emb[:, 0]),
            pl.Series("tsne_y", _tsne_emb[:, 1]),
        ]
    )
    return (tsne_df,)


@app.cell
def _(build_scatter_plot, chem_run_button, mo, root_name, tsne_df):
    mo.stop(not chem_run_button.value)

    step_limit = int(tsne_df["step"].max()) if not tsne_df.is_empty() else 1

    # Plot t-SNE
    _tsne_chart = build_scatter_plot(
        tsne_df,
        "tsne_x",
        "tsne_y",
        f"t-SNE of Hit Fingerprints for {root_name}",
        step_limit,
    )

    save_tsne_button = mo.ui.run_button(label="Save t-SNE plots")
    mo.vstack([mo.ui.altair_chart(_tsne_chart), save_tsne_button])
    return (save_tsne_button,)


@app.cell
def _(mo, root_path, save_tsne_button, tsne_chart):
    mo.stop(not save_tsne_button.value)
    tsne_chart.save(root_path / "tsne_diverse_hits.html")
    tsne_chart.save(root_path / "tsne_diverse_hits.svg")
    tsne_chart.save(root_path / "tsne_diverse_hits.png", scale_factor=2.0)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
