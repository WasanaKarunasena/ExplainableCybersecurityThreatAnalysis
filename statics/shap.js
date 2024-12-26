function createShapBarChart(featureNames, shapValues, containerId) {
    const data = [
        {
            x: shapValues,
            y: featureNames,
            type: 'bar',
            orientation: 'h',
            marker: {
                color: shapValues.map((value) => (value > 0 ? 'red' : 'blue')),
            },
        },
    ];

    const layout = {
        title: 'SHAP Values for Features',
        xaxis: { title: 'SHAP Value' },
        yaxis: { title: 'Feature', automargin: true },
    };

    Plotly.newPlot(containerId, data, layout);
}
