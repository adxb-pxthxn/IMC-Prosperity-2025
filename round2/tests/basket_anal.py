import marimo

__generated_with = "0.12.7"
app = marimo.App(width="full", app_title="Analysis")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Analysis of Basket Data""")
    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    return pd, plt


app._unparsable_cell(
    r"""
    data=
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
