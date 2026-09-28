# M3 pilot status

The multiscale temporal model and one-epoch smoke product are implemented.
The 30-epoch pilot was started but stopped after its first configuration
showed that the current full-grid recurrent training loop is too slow for this
mechanism. The smoke product is retained under `development_v0/m3_smoke/` and
is not a scientific result. M3 needs a separate training-speed refactor before
its pilot metrics are collected.
