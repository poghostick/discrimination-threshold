# Discrimination Threshold

This is the accompanying Github repository for the [Medium blogpost](https://medium.com/towards-data-science/dt-plot-finding-the-optimal-discrimination-threshold-for-binary-classifier-c3f326d6326f). 

## How to Install?

The easiest way to run this code locally is through creating a Python virtual environment and installing all the dependencies from `requirements.txt`.

This is how to do it on Linux:

```python3
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

To leave the created environment just run the command `deactivate`

## How to run?

There are two ways: 

(1) Running from the notebook Discrimination-Threshold.ipynb, by starting a Jupyter kernel in the current folder with the command:
```console
$ jupyter notebook
```
In that case the plot will be activated through [JupyterDash](https://medium.com/plotly/introducing-jupyterdash-811f1f57c02e) instance;

(2) Directly from the command line using
```console
$ python app.py
```
and openning the URL address `http://127.0.0.1:8051/` in your browser. In this case, the plot will be activated through the standard [Plotly Dash](https://dash.plotly.com).
