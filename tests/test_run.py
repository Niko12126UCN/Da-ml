import pytest

import run


@pytest.mark.skipif(run.port_in_use(run.API_HOST, run.API_PORT) or run.port_in_use(run.UI_HOST, run.UI_PORT),
                    reason="the app is already running on this PC (ports 5001/8050 in use)")
def test_run_check_starts_and_stops_both_servers():
    assert run.main(["--check"]) == 0
    assert not run.port_in_use(run.API_HOST, run.API_PORT)
    assert not run.port_in_use(run.UI_HOST, run.UI_PORT)
