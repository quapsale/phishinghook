from critdd import Diagrams
import numpy as np
import pandas as pd

df = pd.read_csv('../data/cdd_data.csv')

# Construct a sequence of CD diagrams
treatment_names = df["method"].unique()
diagram_names = df["diagram"].unique()
Xs = [] # collect an (n,k)-shaped matrix for each diagram
for n in diagram_names:
    diagram_df = df[df.diagram == n].pivot(
        index = "dataset",
        columns = "method",
        values = "metric"
    )[treatment_names] # ensure a fixed order of treatments
    Xs.append(diagram_df.to_numpy())
two_dimensional_diagram = Diagrams(
    np.stack(Xs),
    diagram_names = diagram_names,
    treatment_names = treatment_names,
    maximize_outcome = False
)

# Customize the style of the plot and export to PDF
two_dimensional_diagram.to_file(
    "2d_diagram.pdf",
    preamble = "\n".join([ # colors are defined before \begin{document}
        "\\definecolor{color1}{HTML}{84B818}",
        "\\definecolor{color2}{HTML}{D18B12}",
        "\\definecolor{color3}{HTML}{1BB5B5}",
        # "\\definecolor{color4}{HTML}{F85A3E}",
        # "\\definecolor{color5}{HTML}{4B6CFC}",
    ]),
    axis_options = { # style the plot
        "cycle list": ",".join([ # define the markers for treatments
            "{color1,mark=*}",
            "{color2,mark=diamond*}",
            "{color3,mark=triangle,semithick}",
            # "{color4,mark=square,semithick}",
            # "{color5,mark=pentagon,semithick}",
        ]),
        "width": "\\axisdefaultwidth",
        "height": "0.75*\\axisdefaultheight",
        "title": "Critical Difference Diagram of Model Scalability"
    },)
