"""A source bank should preserve a recoverable conditional residual response."""
import numpy as np

from river_graph.models.doc_source_retrieval import SourceResidualBank
from river_graph.models.source_response_profiles import SourceResponseBank


def test_seasonal_source_response_survives_shrinkage_and_serialization():
    rng = np.random.default_rng(43)
    months = np.arange("2000-01", "2005-01", dtype="datetime64[M]")
    n, t = 8, len(months)
    hydro = rng.uniform(.5, 2, (n, t, 2))
    ecology = rng.normal(size=(n, 9))
    cells = np.arange(n*t)
    context = np.full(n*t, 4.)
    phase = 2*np.pi*(months.astype(int) % 12)/12
    truth = context+np.tile(1.8*np.sin(phase), n)
    kwargs = {"station_names": [f"s{i}" for i in range(n)]}
    old = SourceResidualBank().fit(ecology, hydro, np.ones_like(hydro, bool), months, cells, context, truth, **kwargs)
    new = SourceResponseBank().fit(ecology, hydro, np.ones_like(hydro, bool), months, cells, context, truth, **kwargs)
    query = {"station_names": ["external"]*t}
    arguments = (np.tile(ecology[0], (t, 1)), hydro[0], np.ones_like(hydro[0], bool), months,
                 context[:t], np.zeros((t, 64)))
    old_error = np.abs(old.query(*arguments, **query)["values"][..., 0].mean(1)*old.residual_scale_-(truth[:t]-4)).mean()
    inputs = new.query(*arguments, **query)
    new_error = np.abs(inputs["values"][..., 0].mean(1)*new.residual_scale_-(truth[:t]-4)).mean()
    assert new_error < .3*old_error
    restored = SourceResidualBank.from_dict(new.to_dict())
    np.testing.assert_array_equal(inputs["values"], restored.query(*arguments, **query)["values"])
