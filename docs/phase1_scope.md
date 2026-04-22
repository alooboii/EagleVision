# Phase 1 Scope

What Phase 1 does:

- builds a geometry-first indoor round-trip training environment
- adapts a frozen Depth Anything V2 baseline with a light residual head
- evaluates both downstream geometric usefulness and direct depth quality
- provides debug panels, configs, tests, and modular code for extension

What Phase 1 does not claim:

- final novel-view synthesis quality on large motions
- generative completion of disoccluded content
- learned fusion or refinement as a contribution
- full end-to-end source-depth prediction training as the main regime

The purpose of this phase is to establish a reliable research backbone for making the depth estimator more geometrically useful.
