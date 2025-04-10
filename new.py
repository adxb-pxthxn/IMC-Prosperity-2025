import marimo

__generated_with = "0.12.7"
app = marimo.App(width="medium")


@app.cell
def _():
    print('Hello')
    return


@app.cell
def _():
    import pandas as pd

    pd.read_csv('combined_data.csv',sep=';')
    return (pd,)

@app.cell
def _():
    print('Hello World this is funny')



if __name__ == "__main__":
    app.run()
