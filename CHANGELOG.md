# Changelog

All notable changes to ∇ TorchEBM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.2] - 2026-08-30

### Added

- [`d382bb2`](https://github.com/soran-ghaderi/torchebm/commit/d382bb2d2c4f79abecfdae7cceacd389be3a9d7f) - explicit y conditioning passthrough on loss forwards (#261)
- [`6598212`](https://github.com/soran-ghaderi/torchebm/commit/6598212b588f3aade979a5ab8621fde021c77661) - explicit y conditioning passthrough on loss forwards (#261)
- [`665003b`](https://github.com/soran-ghaderi/torchebm/commit/665003b274f343701de5df8f8277e25661ecfe72) - classifier-free guidance label dropout with pluggable null condition (#262)
- [`99a073c`](https://github.com/soran-ghaderi/torchebm/commit/99a073cfc3d3a03c395ea3568a85ac5fce59643d) - batched classifier-free guidance wrapper usable by every sampler (#262)
- [`77b0e3c`](https://github.com/soran-ghaderi/torchebm/commit/77b0e3cb624a66493549c7a9558a430f4b29a9d2) - explicit y conditioning passthrough on sample (#262)
- [`1e64121`](https://github.com/soran-ghaderi/torchebm/commit/1e641219d8ae40d087476a2da4c2d8a0498b141a) - probe that the model consumes y on the first conditional call (#263)
- [`620d493`](https://github.com/soran-ghaderi/torchebm/commit/620d4931c1a04008d232e9d5423f4fb820cb863b) - standard conditional flow matching loss (#268)
- [`ba785fe`](https://github.com/soran-ghaderi/torchebm/commit/ba785fe580d0457a292f9213a71f43f19cc6adc5) - reject unknown constructor kwargs with supported params and version (#269)

### Changed

- [`eb090ef`](https://github.com/soran-ghaderi/torchebm/commit/eb090ef68e85e8e838849b43fd35a9ab34d4f88a) - shared internal base for interpolant losses (#266)
- [`09d532a`](https://github.com/soran-ghaderi/torchebm/commit/09d532ae9d9663f09bb32b1d73111c400ae8291e) - shared internal base for interpolant losses (#266) + t sampler and per-timestep weight hook on energy matching (#267)
- [`47d8e3c`](https://github.com/soran-ghaderi/torchebm/commit/47d8e3ccc06aacef985ab29fed5c86ed436188e3) - shared internal base for interpolant losses (#266)

## [0.8.0] - 2026-08-28

### Added

- [`e9a0aaa`](https://github.com/soran-ghaderi/torchebm/commit/e9a0aaab45584d4986d23e3205d5993b75e0e37d) - add guarded process-group helpers + gloo spawn harness and fsdp2 score-matching pins
- [`767ba76`](https://github.com/soran-ghaderi/torchebm/commit/767ba76f695c7326249a75e1237060378d601ae6) - add guarded process-group helpers + gloo spawn harness and shim tests #243
- [`06f0993`](https://github.com/soran-ghaderi/torchebm/commit/06f099305c21ad95aee348a9992cec71e0daabbb) - functional score path for sharded dtensor models + tests for functional score parity and fsdp2 double-backward pins(#244)
- [`f1ba1de`](https://github.com/soran-ghaderi/torchebm/commit/f1ba1de2929bcf8ef79f223c8c50840a32d0ed6a) - thread generator through sampler, integrator, loss, coupling,  rng + generator contract and per-rank decorrelation tests
- [`0036630`](https://github.com/soran-ghaderi/torchebm/commit/003663034ec523d3824fd34687b63819559cc5eb) - unsharded context manager for k-step mcmc under fsdp2 + energy model parity tests #246
- [`bbef1b8`](https://github.com/soran-ghaderi/torchebm/commit/bbef1b83efa7630f1190fffa980132298be9e470) - opt-in cross-rank pcd replay buffer mixing + rank-local semantics and checkpoint contract
- [`0211394`](https://github.com/soran-ghaderi/torchebm/commit/02113943cee5dae92066b8d8aa21c14ab81f42fa) - process-group-aware global-batch sinkhorn coupling with rank0-broadcast draws
- [`fdfe71b`](https://github.com/soran-ghaderi/torchebm/commit/fdfe71baaaa928099243c7980b0bf916a488dc52) - fail-fast second-order guards under sharded params + fsdp2 loss parity matrix (#248)
- [`2ac1a61`](https://github.com/soran-ghaderi/torchebm/commit/2ac1a618688b1a84ef816d2b6c31a67f996bcb84) - pluggable c(t) weight family with constant flow matching endpoint (#254)
- [`91a824c`](https://github.com/soran-ghaderi/torchebm/commit/91a824ce2834dfee8c2b092c2179710eb75185d6) - probe that the model consumes t when time_invariant is false (#255)
- [`27d659e`](https://github.com/soran-ghaderi/torchebm/commit/27d659e81e8958db7598bdbf5bf5f1c9fe882686) - edm lognormal t sampler and per-timestep loss weight hook (#256)

### Changed

- [`aac6092`](https://github.com/soran-ghaderi/torchebm/commit/aac6092fa57594cd378ba1d8cbc0899a65aacde3) - point tests and docs at the relocated utils shim
- [`f6a162b`](https://github.com/soran-ghaderi/torchebm/commit/f6a162b8da74838246d52c7ce919d77d4f605736) - drop unused device and dtype args from _resolve_diffusion

### Fixed

- [`56e6522`](https://github.com/soran-ghaderi/torchebm/commit/56e65226f7c2a631c7dd9c154e30f5fbefed0aa6) - numpy-free tensor broadcasts + optimizer-ready grads on the functional score path (#249)
- [`c12a58f`](https://github.com/soran-ghaderi/torchebm/commit/c12a58fbf9efb70c85aae7f623f049f9dbaa0fd3) - numpy-free tensor broadcasts + optimizer-ready grads on the functional score path, shim moved to utils (#249)
- [`ab991d3`](https://github.com/soran-ghaderi/torchebm/commit/ab991d3877c84a70ad4d451760111b5819b088ed) - reject unknown constructor kwargs with supported params and version (#253)
- [`dab078a`](https://github.com/soran-ghaderi/torchebm/commit/dab078a3dfed9f4e271261236470bad517f7c81c) - condition eqm fields on zeroed time and negated velocity score in flowsampler (#255)

### Performance

- [`4eb623f`](https://github.com/soran-ghaderi/torchebm/commit/4eb623f69729348d680f245b41cc2895ad5fbd5b) - lazy-load external badges and drop dead analytics snippet
- [`5c66fba`](https://github.com/soran-ghaderi/torchebm/commit/5c66fba001f83289aff8d5f1200d51e9d718f5d3) - lazy-load external badges and drop dead analytics, load mathjax and mermaid on demand

### Other

- [`a25471b`](https://github.com/soran-ghaderi/torchebm/commit/a25471b934eaccedfcea13098a39483828d5ac89) - functional score parity and fsdp2 double-backward pins
- [`f583901`](https://github.com/soran-ghaderi/torchebm/commit/f583901d2fbebb4eae9ea8a0c5ae26a3acc7691b) - require python 3.10 and torch 2.10 for generator support
- [`80e1bce`](https://github.com/soran-ghaderi/torchebm/commit/80e1bce94affbb57174eb07405b02f884e779bd0) - pin sync-free hot paths and document the gpu-first contract #239
- [`264a73f`](https://github.com/soran-ghaderi/torchebm/commit/264a73fc990b824290dbb8ab7d93c5d46b426c25) - standard software title, unified bibtex, orcid and preferred-citation
- [`86a5c6c`](https://github.com/soran-ghaderi/torchebm/commit/86a5c6c603132980459c62a630cd52655fa43acd) - pin ema updates on identically sharded params and the dcp + rank-local checkpoint recipe
- [`3499d43`](https://github.com/soran-ghaderi/torchebm/commit/3499d435d4a0082f997d70c9f61d8ec17a0af143) - device-generic nccl harness and 4-gpu fsdp2 training-step benchmark (#249)
- [`b9c6030`](https://github.com/soran-ghaderi/torchebm/commit/b9c603093abfd4475afa0b5a2bba7653725fa36a) - regenerate unreleased on merges and promote on releases via git-cliff
- [`7ee7cfd`](https://github.com/soran-ghaderi/torchebm/commit/7ee7cfdc85679e912512af7592a24707a161e30a) - regenerate 0.7.1, 0.7.5, and unreleased sections from commit history
- [`1e0fc43`](https://github.com/soran-ghaderi/torchebm/commit/1e0fc43bf062f227d0e30f9a54fb725c88d28ab0) - pin sde integration through the noise_scale path
- [`3a84ff6`](https://github.com/soran-ghaderi/torchebm/commit/3a84ff616f1f6b1dd2631956f98944e266a89e01) - single rendered bibtex with a drift check against citation.cff

## [0.7.5] - 2026-07-16

### Added

- [`7d7cfb2`](https://github.com/soran-ghaderi/torchebm/commit/7d7cfb235049a4f0a709ad0cd3d8d6da9d2fe3a7) - add explicit midpoint RK2
- [`fa1263d`](https://github.com/soran-ghaderi/torchebm/commit/fa1263dcfded4a7748ccceed79d0f5b095eccee4) - thread model_kwargs conditioning through all samplers,  losses, base modules, and couplings
- [`96f1268`](https://github.com/soran-ghaderi/torchebm/commit/96f126869ac6da4381350e1af1066e6f5191df70) - eqmenergy adapter for gradient sampling of equilibrium-matching fields
- [`51a4ded`](https://github.com/soran-ghaderi/torchebm/commit/51a4ded2b6c7e49f088e7f04c2e952bd564f6f1f) - x0 and coupling support in equilibriummatchingloss

### Changed

- [`c800f26`](https://github.com/soran-ghaderi/torchebm/commit/c800f268d5fcc199b0badbc105c78d0a5e9b9866) - generate synthetic datasets with torch instead of numpy

### Fixed

- [`60284c5`](https://github.com/soran-ghaderi/torchebm/commit/60284c52c2631630414d21def5cecc7c701dc93d) - self-host paper figures, semanticscholar serves non-image content-type that github/pypi camo rejects
- [`b395194`](https://github.com/soran-ghaderi/torchebm/commit/b395194383d2f90c0c0aad706d9ba40b5a94aaac) - register EnergyMatchingLoss model_type so its bench builds

### Performance

- [`0496a12`](https://github.com/soran-ghaderi/torchebm/commit/0496a12c0e93b40e92d3da850fe370e636036aa7) - instant navigation, deferred pinned mathjax, cdn preconnect
- [`393865d`](https://github.com/soran-ghaderi/torchebm/commit/393865d937ac3b5e6f6ec4dd93c25644cf48b99a) - compute basemodel.gradient in the input dtype with fp32 opt-in + device-tensor metrics, sync once per epoch
- [`6231624`](https://github.com/soran-ghaderi/torchebm/commit/623162453117907660c88209937a199ac56c1a9e) - drop per-step scalar-tensor syncs; single greedy transfer

### Other

- [`e85e5a7`](https://github.com/soran-ghaderi/torchebm/commit/e85e5a7ae79f7e1809783c3e91db48b4422ef61e) - expand midpoint coverage
- [`879b35d`](https://github.com/soran-ghaderi/torchebm/commit/879b35dbe5a10358840a59ba944202b3ffa0da6c) - add contributing guide
- [`11f42fc`](https://github.com/soran-ghaderi/torchebm/commit/11f42fcab7c046ae8e73a9f9e504dd73a015b32d) - cover model_kwargs contract, conditional negatives, deprecations over all components and couplings!
- [`e3e2c03`](https://github.com/soran-ghaderi/torchebm/commit/e3e2c0362c924bf524a09bee6a32dbffee7b4654) - cover eqmenergy, x0/coupling, and sampling-route agreement
- [`c6be616`](https://github.com/soran-ghaderi/torchebm/commit/c6be6165b2384a96471f8aa94e5f8f37e3a3d798) - coherent eqm sampling routes and diverse coda
- [`8044140`](https://github.com/soran-ghaderi/torchebm/commit/80441402f2e712d0f935c7d692f4d3f9f2e34811) - lead with simulation-free, gpu-first narrative across pypi, docs, and readme
- [`e5ce988`](https://github.com/soran-ghaderi/torchebm/commit/e5ce988dfb81edf8444f7cb7ae67824bad074261) - drop verbose inline comments from library code

## [0.7.1] - 2026-07-13

### Added

- [`e8fd25d`](https://github.com/soran-ghaderi/torchebm/commit/e8fd25df9627dd516962c2bead4fa1a32f9c65f3) - auto-generate example pages from examples/ via in-memory mkdocs hooks
- [`55a2705`](https://github.com/soran-ghaderi/torchebm/commit/55a2705270c825eff185f96fad1ee7f08e6060f3) - add basesymplecticintegrator base for leapfrog family
- [`dc0c5c3`](https://github.com/soran-ghaderi/torchebm/commit/dc0c5c3a6c5636158772eaf896d2ae7e839c775e) - add get_integrator string registry and resolver
- [`90cfb00`](https://github.com/soran-ghaderi/torchebm/commit/90cfb00f2f9952ec47c2212a9936c0a7bbc94a00) - accept integrator instances and registry names in langevin/hmc/rmhmc
- [`ad7b274`](https://github.com/soran-ghaderi/torchebm/commit/ad7b2748f8bebbe0f69c1b27c08a248301ad008f) - add temperaturescheduler and basecoupling primitives
- [`3c2c8cc`](https://github.com/soran-ghaderi/torchebm/commit/3c2c8cc69829ee898bad3cab2580318ca479a2b9) - constructor integrator for flowsampler, deprecate per-call method args, drop torchdiffeq
- [`b3931cc`](https://github.com/soran-ghaderi/torchebm/commit/b3931cc27f33966e6fa0327eb2c74d67207fc3cc) - structured couplingresult return and generalized root contract
- [`97ad838`](https://github.com/soran-ghaderi/torchebm/commit/97ad8381db0bf532bbf9d2ef261cc4d4e1798f0a) - greedy, unbalanced sinkhorn, and model-induced reflow couplings
- [`a92542f`](https://github.com/soran-ghaderi/torchebm/commit/a92542f2ea9e1246a765885ef83958bf21bc9102) - wire energymatchingloss exports and flow helpers
- [`698c02c`](https://github.com/soran-ghaderi/torchebm/commit/698c02c463d77a5c628f2426bdf858c2024544ae) - interaction model for energy matching repulsion
- [`6138091`](https://github.com/soran-ghaderi/torchebm/commit/6138091307e1e3c4084ffe7cd1d21fbf51663605) - flowsampler mode-based constructor and conforming sample contract (#171)
- [`146a737`](https://github.com/soran-ghaderi/torchebm/commit/146a73760958ca38b74c368cba783b2fdc1de7df) - tiered ci-tested curriculum with generated pages
- [`0142c5d`](https://github.com/soran-ghaderi/torchebm/commit/0142c5d43fbed1d1d80bf49876d33e938bb69183) - concepts section, generated component views, and new navigation. slim the sdist, add pyyaml to dev, guard package discovery

### Changed

- [`aaef0ed`](https://github.com/soran-ghaderi/torchebm/commit/aaef0ed2dce81b8a35fa61fac29ad9cc1894ce54) - move get_interpolant to interpolants and add resolve_interpolant
- [`3e753fe`](https://github.com/soran-ghaderi/torchebm/commit/3e753fe510ec366a2f811a9ff380c8a7359e01c8) - int-or-tuple dim with shared state init and drop kwargs cascades (#172)
- [`c8b1d6c`](https://github.com/soran-ghaderi/torchebm/commit/c8b1d6cc6c6ad915d784b0d0835241ee7c006daa) - export resolve_integrator and validate family for registry names (#174)

### Fixed

- [`df8b33d`](https://github.com/soran-ghaderi/torchebm/commit/df8b33d42f86a4079a93de040d2eb0390d7d8d93) - add repo root to pytest pythonpath so tests.conftest imports survive package discovery guard

### Other

- [`81feb5b`](https://github.com/soran-ghaderi/torchebm/commit/81feb5bce12275ad3fd1d42b0822f064187d6ca7) - remove polyfill to avoid its popup on the website
- [`cc5dc19`](https://github.com/soran-ghaderi/torchebm/commit/cc5dc198c7ab2e66e4b47c7cfc7dce270852d537) - cover symplectic base, registry, integrator injection
- [`0cba390`](https://github.com/soran-ghaderi/torchebm/commit/0cba3907a3e1e8934dd08c06faf97b1c737e9a38) - euler/dopri5 variants via ctor integrator, changelog
- [`3237d7f`](https://github.com/soran-ghaderi/torchebm/commit/3237d7f4385fb11ed720afafdf5d179f424619a1) - update issue templates and changelog
- [`75c4d47`](https://github.com/soran-ghaderi/torchebm/commit/75c4d47027acfca487a0609e9a280df0e809db0a) - cross-sampler api contract guard (#175)
- [`9de9916`](https://github.com/soran-ghaderi/torchebm/commit/9de9916ff5d390fc0016be453e18ca7c9040ebf8) - update changelog for flowsampler standardization (#170)
- [`84ae09a`](https://github.com/soran-ghaderi/torchebm/commit/84ae09a3264374eb8a63a4323352581723455d11) - readme rewrite and 0.7.1 release notes

## [0.6.0] - 2026-05-31

### Added

- [`3dcb39d`](https://github.com/soran-ghaderi/torchebm/commit/3dcb39d810a506730ac288a2ce9069e9d072bdcb) - add RiemannianManifoldHMC sampler and related tests
- [`f12fe61`](https://github.com/soran-ghaderi/torchebm/commit/f12fe613893a6ee8acbe00bb03e48e898993df73) - implement GeneralisedLeapfrogIntegrator for non-separable Hamiltonian dynamics
- [`0360dff`](https://github.com/soran-ghaderi/torchebm/commit/0360dff9ee8ea255b3e28d7f9145e4e27aeb48fa) - add BackwardEulerMaruyamaIntegrator and update related tests
- [`3802a9f`](https://github.com/soran-ghaderi/torchebm/commit/3802a9f6ccf552d31af3508c68a32e6a7b050158) - add support for implicit RK integrators in the base RK SDE class

### Changed

- [`4d649c2`](https://github.com/soran-ghaderi/torchebm/commit/4d649c2895d155c99b194123a4a5392778fc7efa) - change diagnostics to `dict[str, tensor]` with named keys (mean, var, energy, acceptance_rate)

### Performance

- [`ac75c63`](https://github.com/soran-ghaderi/torchebm/commit/ac75c6327cb2bfc235b7c1fc8a35896621372b35) - addcmul fusion + safe clamp + cache metric Cholesky + stiff-metric tests

## [0.5.8] - 2026-02-17

### Added

- [`92617fe`](https://github.com/soran-ghaderi/torchebm/commit/92617feeda13427aa923496f9556e07a38e39581) - add Bosh3Integrator with adaptive step-size control and FSAL property
- [`11808de`](https://github.com/soran-ghaderi/torchebm/commit/11808de9a7d1b7134dfca43d4cfec4fc6ff9d9c7) - implement RK4 integrator with fixed step and Butcher tableau
- [`392379e`](https://github.com/soran-ghaderi/torchebm/commit/392379e7a1cef21db38932f2f5ac37b3013be6ec) - add Adaptive Heun integrator with embedded error estimation

### Other

- [`705ac02`](https://github.com/soran-ghaderi/torchebm/commit/705ac02b6b202bc161751ffcf167333a0dbeaa6d) - Add comprehensive tests for RK4Integrator functionality
- [`fc4184c`](https://github.com/soran-ghaderi/torchebm/commit/fc4184ca414f30eaf8fecc04b6d6a310eaf38ddc) - enhance documentation for Dopri5 and Dopri8 integrators with additional equations and references
- [`8d8f0a3`](https://github.com/soran-ghaderi/torchebm/commit/8d8f0a3c56a5c1e3c0fe8f9c64c792c69aea8b07) - update navigation structure and include new integrator classes

## [0.5.7] - 2026-02-12

### Changed

- [`28d0fcb`](https://github.com/soran-ghaderi/torchebm/commit/28d0fcb2de2337cdf1df79b1ad68867fb296e429) - consolidate Dopri5 and Dopri8 integrators into a single module
- [`dcd2cbe`](https://github.com/soran-ghaderi/torchebm/commit/dcd2cbee6da17692fc15987a3d4ea64763c2ceed) - streamline imports across modules and enhance __all__ exports

### Other

- [`f40e0c3`](https://github.com/soran-ghaderi/torchebm/commit/f40e0c30200fbd8c0faec0af14b6149d058767f9) - format mathematical equations in the Equilibrium Matching loss documentation
- [`49a01bf`](https://github.com/soran-ghaderi/torchebm/commit/49a01bf721bebcc5f214ea788645df5dfa4b3221) - all tests passing for dopri8

## [0.5.6] - 2026-02-12

### Added

- [`57d8e8f`](https://github.com/soran-ghaderi/torchebm/commit/57d8e8f2486f77daa946fa6e889e150c4188c6dd) - add Dopri5Integrator class with adaptive step-size control and Butcher tableau
- [`dbf7890`](https://github.com/soran-ghaderi/torchebm/commit/dbf7890a0d12b8bbfe808058fa108ec65aadd66e) - enhance BaseRungeKuttaIntegrator with max_step_size and norm parameters; update EulerMaruyama and Heun integrators to reflect changes
- [`b9b98bf`](https://github.com/soran-ghaderi/torchebm/commit/b9b98bf451c521747d1528277a6261d7bf9a87db) - update integrators to use explicit drift functions and refactor related tests
- [`e4d1182`](https://github.com/soran-ghaderi/torchebm/commit/e4d118297d380ec866c24fc142cba5f76a102ed7) - implement adaptive step-size control in BaseRungeKuttaIntegrator and update integrators
- [`dc25fb7`](https://github.com/soran-ghaderi/torchebm/commit/dc25fb7a27d4d736028d3f8b7e85a1e9004cf5e0) - add BaseRungeKuttaIntegrator class and update imports

### Changed

- [`7eeea77`](https://github.com/soran-ghaderi/torchebm/commit/7eeea77217a9012eabc0ac8f9c5c0933dc068156) - remove unused imports from integrators module
- [`18a419d`](https://github.com/soran-ghaderi/torchebm/commit/18a419ddbdbed928d83eb414670016ba114af386) - integrators to remove deprecated `model` parameter
- [`4282f27`](https://github.com/soran-ghaderi/torchebm/commit/4282f2701b41edc46e6028224f8bd02107f5711a) - Refactor documentation structure and remove LLM export scripts

### Other

- [`0e442bc`](https://github.com/soran-ghaderi/torchebm/commit/0e442bca10e62e06457ef0c448c66077a30c2aee) - Dopri5Integrator tests
- [`0a951a2`](https://github.com/soran-ghaderi/torchebm/commit/0a951a2a7d7207bdfb6b33d4045e36a71792a9aa) - remove LLM documentation generation step from CI workflow
- [`3b9a3d7`](https://github.com/soran-ghaderi/torchebm/commit/3b9a3d7b2a6f5def105d817aec789505ac63f428) - add new animation assets for Gaussian and circles flow

## [0.5.4] - 2026-02-11

### Fixed

- [`0fd292c`](https://github.com/soran-ghaderi/torchebm/commit/0fd292c46e60470f8664c9f2f46c7235f7b296c4) - The EqM loss was using a hardcoded target (x0 - x1) * c(t), which is only correct for the Linear interpolant. Changed the target  to -ut * ct, which correctly uses each interpolant's velocity schedule.
- [`1c6741d`](https://github.com/soran-ghaderi/torchebm/commit/1c6741d4d4aaab06e213b1cde70dd4956d050b5c) - update tensor dimensionality checks to require length >= 2

### Other

- [`ffab8d7`](https://github.com/soran-ghaderi/torchebm/commit/ffab8d744a47560fbcf2e407b1f408147db89485) - update installation command for testing package to include development dependencies
- [`793a2c4`](https://github.com/soran-ghaderi/torchebm/commit/793a2c4d80ade63aa05e103849acf7ce840808f8) - update .gitignore and improve Equilibrium Matching loss documentation
- [`b102503`](https://github.com/soran-ghaderi/torchebm/commit/b102503b307aa177fe1e9a9875416080410c4a49) - enhance documentation for Equilibrium Matching and add integration tests for single-step scenarios
- [`a2218e5`](https://github.com/soran-ghaderi/torchebm/commit/a2218e58c3a3f10ef54e5f4a6630cc42ddb5daff) - add LLM documentation generation and update workflows

## [0.5.1] - 2025-12-29

### Added

- [`401a2e8`](https://github.com/soran-ghaderi/torchebm/commit/401a2e8c9b5777bdcd24987654facdc04a656693) - reorganize utility functions into dedicated modules and enhance test coverage
- [`d1bc466`](https://github.com/soran-ghaderi/torchebm/commit/d1bc466cdf3ff1824adca564d91b8b56c18ecce4) - add support for multiple diffusion forms in FlowSampler and enhance documentation
- [`8ca6752`](https://github.com/soran-ghaderi/torchebm/commit/8ca67524a1fb42bcf6fad4bfc4c9b6160673c4b8) - add custom parameters for c(t) in EquilibriumMatchingLoss and update related tests
- [`8bf0d7b`](https://github.com/soran-ghaderi/torchebm/commit/8bf0d7b407c2ddf53a0331702a928a492869004d) - enhance diffusion form handling in the base_interpolant with additional options and improved error messaging
- [`2eb3494`](https://github.com/soran-ghaderi/torchebm/commit/2eb3494a4142b47a32b5c926ef8e2d6711ef198a) - enhance EquilibriumMatchingLoss with explicit energy formulations and detailed docstring updates

### Fixed

- [`87f4f08`](https://github.com/soran-ghaderi/torchebm/commit/87f4f08be45ee68bd7cb7036429ebdae64beb2bf) - correct save_checkpoint call by including optimizer parameter

### Other

- [`0ec4e4a`](https://github.com/soran-ghaderi/torchebm/commit/0ec4e4aa06296026afd820a160c21966e1635614) - Update torchebm/losses/equilibrium_matching.py
- [`e9a443d`](https://github.com/soran-ghaderi/torchebm/commit/e9a443d5873357a95687fbd1a3f8814999853836) - Update torchebm/losses/equilibrium_matching.py

## [0.5.0] - 2025-12-23

### Added

- [`9171071`](https://github.com/soran-ghaderi/torchebm/commit/91710710485456a5df59a5945b6bc731acac8ead) - introduce models subpackage
- [`eab4cf2`](https://github.com/soran-ghaderi/torchebm/commit/eab4cf294a062a87729c053fd707cb97a527280c) - add all new features. integrator, intreplants, losses, samplers
- [`546b241`](https://github.com/soran-ghaderi/torchebm/commit/546b241cdff25e845ddbfa052712c47742428ea1) - add new samplers. flow, gd, nestrov gd.
- [`59ecde9`](https://github.com/soran-ghaderi/torchebm/commit/59ecde960ebc66fff7a1f5b056a76e7886c78752) - add interpolants for flow matching. linear, cosine, and variance preserving
- [`973b5d1`](https://github.com/soran-ghaderi/torchebm/commit/973b5d11aa4427408409af2d6928c9e151d8eeb1) - introduce new integrators with standard API. heun, euler-maruyama, leapfrog, integrator utils
- [`0825618`](https://github.com/soran-ghaderi/torchebm/commit/08256186a8dd6558e84dd3b1e6f030b1c3504d2c) - init base_integrator and base_interpolant
- [`9df60af`](https://github.com/soran-ghaderi/torchebm/commit/9df60afed010039ba2d88eb442f3c314ebf003cd) - hmc simplicfied and use integrators API.
- [`7417bb4`](https://github.com/soran-ghaderi/torchebm/commit/7417bb4830e1b2c103a26d5e5b2d242b8c8b6d88) - integrators package resolves [#88](https://github.com/soran-ghaderi/torchebm/pull/88) [#89](https://github.com/soran-ghaderi/torchebm/pull/89)

### Changed

- [`e9ae0ce`](https://github.com/soran-ghaderi/torchebm/commit/e9ae0ce44868586655b6e09fc343c70792305488) - remove obsolete files and make base_sampler more flexible
- [`046e166`](https://github.com/soran-ghaderi/torchebm/commit/046e16695b8cf88548e142348bba2992faeebb65) - move euler and leapfrog integrators to dedicated files under integrators directly
- [`f7cd8b6`](https://github.com/soran-ghaderi/torchebm/commit/f7cd8b6c43fc8fcd915c18a21750f90095cfef94) - centralize mixed-precision and other minor changes
- [`7d4f1a3`](https://github.com/soran-ghaderi/torchebm/commit/7d4f1a3a4112695ac2ddd96fc655d499f8661d90) - use standard notations. model instead of energy_function acrros the library

### Fixed

- [`8132dbe`](https://github.com/soran-ghaderi/torchebm/commit/8132dbeabd13dad436285720970dd1b18c1fe678) - remove doctest split option from doc gen.py and remove example located in  tests
- [`f74ff74`](https://github.com/soran-ghaderi/torchebm/commit/f74ff74f5159f38884e5d76e6d4b71e3336bc195) - attempt 1 fix the test errors
- [`9fcaa64`](https://github.com/soran-ghaderi/torchebm/commit/9fcaa64b6bdb930ab71668e1dfb1e1bce98fa34a) - update visualization.py
- [`aaecb85`](https://github.com/soran-ghaderi/torchebm/commit/aaecb85d161cff0ccecf7ca5161b324f6d6cc8f1) - fix ci/cd issues and update forms

### Other

- [`257053b`](https://github.com/soran-ghaderi/torchebm/commit/257053b771685e67c1eb6984b6c151c2355afbf2) - Update tests/losses/test_equilibrium_matching.py
- [`ebf79ec`](https://github.com/soran-ghaderi/torchebm/commit/ebf79ec50e3144dc4d78a9f502c60b6ba0598ef1) - pass tests for samplers
- [`da560fc`](https://github.com/soran-ghaderi/torchebm/commit/da560fc98b5596f1c2f723677905b3f6970980fc) - pass tests for losses
- [`e248839`](https://github.com/soran-ghaderi/torchebm/commit/e2488396554664260cd74137d0ce9fb001f7e679) - add assets for docs
- [`ce25b2f`](https://github.com/soran-ghaderi/torchebm/commit/ce25b2fe3717b2c67665bb60908bb755ebc0ca19) - all tests for interpolants are passing.
- [`275189a`](https://github.com/soran-ghaderi/torchebm/commit/275189abf502b2b211ec21c76e2a441c7935573d) - add tests for integrators
- [`f3ce91f`](https://github.com/soran-ghaderi/torchebm/commit/f3ce91f1da2c6e21f4450039115b182ef2112ec9) - update mkdocs and remove blog sort by title
- [`06a49d6`](https://github.com/soran-ghaderi/torchebm/commit/06a49d6876d9f6a34f2a799ec6dfd1cf11fa2d24) - update docstrings for integrators, interpolants, samplers, and losses.
- [`4ec2907`](https://github.com/soran-ghaderi/torchebm/commit/4ec29071f8882a476f389d43b5eab762ca380a36) - update the nav bar and minor modifications
- [`4f001c8`](https://github.com/soran-ghaderi/torchebm/commit/4f001c8f7d9731907fd11ef4f6fb0e1d1fab0928) - update hmc and langevin wrt to new integrators
- [`e66ec63`](https://github.com/soran-ghaderi/torchebm/commit/e66ec630b6399e64ae106b326ea18f4d9f626047) - run doc build on pr
- [`e7221d4`](https://github.com/soran-ghaderi/torchebm/commit/e7221d407d528cf722c08d3f3255d72690dae739) - fix yaml syntax
- [`41684b6`](https://github.com/soran-ghaderi/torchebm/commit/41684b69fb4f62f9b1f12fa85d520f60b96bd336) - fix version issue
- [`2f3dd1b`](https://github.com/soran-ghaderi/torchebm/commit/2f3dd1bcfb2646d2287077ce6c1dcd5d3c25df00) - install minify
- [`7a02632`](https://github.com/soran-ghaderi/torchebm/commit/7a02632c9890dd094c1260408601027b36b9c3df) - test python from 3.8 to 3.13
- [`1002382`](https://github.com/soran-ghaderi/torchebm/commit/100238288c44f74dbb208620c525a52a7647f846) - add assets used in the docs2
- [`1883b19`](https://github.com/soran-ghaderi/torchebm/commit/1883b1934e6bfe1a1d40dfcc8895660c657e454b) - add assets used in the docs
- [`ddcf596`](https://github.com/soran-ghaderi/torchebm/commit/ddcf596fcfc905150971057084df4c87abbfc0a8) - update website completely
- [`b0531d1`](https://github.com/soran-ghaderi/torchebm/commit/b0531d1f92aea91c2fa02572cd7d928acf91bfe2) - update guides
- [`cc8f615`](https://github.com/soran-ghaderi/torchebm/commit/cc8f61503f36b6d5cbd5ac749db2db62bbf1a5fa) - remove all docstring verbosity. move to blog posts
- [`ea6febe`](https://github.com/soran-ghaderi/torchebm/commit/ea6febee8950ccd62a4b409c1368b004eb7607ec) - untrack generated files and update .gitignore
- [`d8a81f5`](https://github.com/soran-ghaderi/torchebm/commit/d8a81f55ec86278add9ce5796f5c957994d09449) - update documentation wrt this refactored code

## [0.4.0] - 2025-10-10

### Changed

- [`b35707d`](https://github.com/soran-ghaderi/torchebm/commit/b35707df64ad09687daab0ac51618ba2f46a0a3e) - score_matching.py

### Fixed

- [`2c79e9e`](https://github.com/soran-ghaderi/torchebm/commit/2c79e9e37e915adedcadb63c1c37d213619251f1) - update score matching losses: ssm and dsm
- [`2a8358d`](https://github.com/soran-ghaderi/torchebm/commit/2a8358dc559f164277c3f6b2b2eab8ed31e401f9) - denoising score matching polished and corrected
- [`494e096`](https://github.com/soran-ghaderi/torchebm/commit/494e09646940310ceabe4e4b4e1c586b6a1c8e51) - _exact_score_matching polished and corrected
- [`298599d`](https://github.com/soran-ghaderi/torchebm/commit/298599d1d361d239261d18beacf564fff882cf89) - resolve sampler integration with device_mixin.py

### Other

- [`651ac9d`](https://github.com/soran-ghaderi/torchebm/commit/651ac9dc912a6388e88781e832d026afaf91686c) - denoising score matching

## [0.3.17] - 2025-08-23

### Added

- [`a4f0f4b`](https://github.com/soran-ghaderi/torchebm/commit/a4f0f4b0b42761588e0bca4090585af5c6f13a0a) - vectorized sliced score matching
- [`facbc5a`](https://github.com/soran-ghaderi/torchebm/commit/facbc5a59fb8e5ee99345d8d4ebdae82f9bdaf8f) - add DeviceMixin
- [`a34b8a0`](https://github.com/soran-ghaderi/torchebm/commit/a34b8a02a63a6e5447da17ded503fadad7f92286) - create dependabot.yml

### Changed

- [`f70d3bb`](https://github.com/soran-ghaderi/torchebm/commit/f70d3bb710200e0a8bd2a396d927a5e3e22fa747) - remove .to() override from base_energy_function.py

### Fixed

- [`e2b2d59`](https://github.com/soran-ghaderi/torchebm/commit/e2b2d59565c6577057da99c991c6de2559a08405) - the ssm is accurately follows the paper

### Other

- [`d2fbd74`](https://github.com/soran-ghaderi/torchebm/commit/d2fbd74398581fd70208ab5f7580ac6582c73dde) - update docs
- [`899502e`](https://github.com/soran-ghaderi/torchebm/commit/899502ea5862ce6ef966108b232958540a4313a5) - rewrite math eq. in standard latex
- [`a1024b1`](https://github.com/soran-ghaderi/torchebm/commit/a1024b1607ab26f82ea7140163c5f44492fd1a14) - add issue_list.yml template
- [`4941e76`](https://github.com/soran-ghaderi/torchebm/commit/4941e76e6de9ff4095319d2fe296e236d1f227ea) - Update issue template/forms @2
- [`233a936`](https://github.com/soran-ghaderi/torchebm/commit/233a9366d164fb1d2b50a3d36ae852cfea6f40a6) - Update issue template/forms
- [`8bd798f`](https://github.com/soran-ghaderi/torchebm/commit/8bd798ff242da409acb120c53c727cc827284989) - Add issue template/forms
- [`d6fb5f8`](https://github.com/soran-ghaderi/torchebm/commit/d6fb5f8c8427b4d0eb2f4f30a00fc2b36f0fa834) - Update dependabot.yml
- [`5eee895`](https://github.com/soran-ghaderi/torchebm/commit/5eee8950e24bdee0d51e521ac36d77f70d0bd165) - README.md
- [`fbcaa7f`](https://github.com/soran-ghaderi/torchebm/commit/fbcaa7f745cf1fb08e63cef27e43ae745c5f7388) - document schedulers
- [`e259f3e`](https://github.com/soran-ghaderi/torchebm/commit/e259f3ec75bedd17078bcb25561284d895a3aa0c) - Update README.md .
- [`3fe2988`](https://github.com/soran-ghaderi/torchebm/commit/3fe2988597426226e62e9f092eae6e1c4b4cc3ca) - Create CODE_OF_CONDUCT.md

## [0.3.7] - 2025-05-02

### Added

- [`753fe78`](https://github.com/soran-ghaderi/torchebm/commit/753fe782012d6336afbf6efbb832d7ce5f8ba8ee) - sliced score matching class
- [`d8d36ba`](https://github.com/soran-ghaderi/torchebm/commit/d8d36ba3b796474df94e5277513e5e8fe28411f3) - denoising score matching class
- [`cd3c37d`](https://github.com/soran-ghaderi/torchebm/commit/cd3c37da1a109970a431179bbbda29b9db335600) - score matching class
- [`4d2e04d`](https://github.com/soran-ghaderi/torchebm/commit/4d2e04d496a8d88be80e77b3bbeb9d2a96bb56b2) - base class for score matching resolve [#63](https://github.com/soran-ghaderi/torchebm/pull/63)

### Other

- [`ef795c2`](https://github.com/soran-ghaderi/torchebm/commit/ef795c2ee0afc01a25345f2bc33640797a16d40f) - fix examples in guides
- [`b83f162`](https://github.com/soran-ghaderi/torchebm/commit/b83f1629efb26109848c6d28e8cb8e2bd732e457) - update guides
- [`fc9f96e`](https://github.com/soran-ghaderi/torchebm/commit/fc9f96e21db1d3dec3b558b209c8063084fa1027) - pass score matching tests
- [`017245c`](https://github.com/soran-ghaderi/torchebm/commit/017245c2819561d6ce4cfab22e1fdb930ca99f72) - enhanced contrastive div functionality
- [`5f06b08`](https://github.com/soran-ghaderi/torchebm/commit/5f06b08dd0341a62454b3533c5866e93e7f72000) - base sampler get_schedulers

## [0.3.5] - 2025-04-27

### Added

- [`cbd69c3`](https://github.com/soran-ghaderi/torchebm/commit/cbd69c31ae4fdbda393f4498e26f61e9cf53d466) - multistep and warmup schedulers

### Other

- [`51cb2ab`](https://github.com/soran-ghaderi/torchebm/commit/51cb2ab0f9f62fc9e0153089d2faba7e2fd2469d) - comprehensive tests for hmc
- [`19cdf8d`](https://github.com/soran-ghaderi/torchebm/commit/19cdf8dafc1ebce8b4b73b7246b331d2e8e4334f) - comprehensive test for schedulers
- [`6c98550`](https://github.com/soran-ghaderi/torchebm/commit/6c985508e677d8d7337e8558eb880ba0ce99aaf5) - passing all loss tests

## [0.3.4] - 2025-04-24

### Added

- [`4c92b1b`](https://github.com/soran-ghaderi/torchebm/commit/4c92b1b5629cd7e7f2e20ce0abbc7099b3c5215d) - update Langevin and base sampler to support noise/step-size schedulers
- [`7b2895f`](https://github.com/soran-ghaderi/torchebm/commit/7b2895f833ee9178c892a03a93890388fc66822c) - add schedulers for sampler's noise and maybe step size
- [`3e65fe0`](https://github.com/soran-ghaderi/torchebm/commit/3e65fe0b344e5712b15ad8121dcd26cd3476b03e) - base energy sample and quality metrics
- [`b99bd8f`](https://github.com/soran-ghaderi/torchebm/commit/b99bd8fbbf1853a9fc8d865ba6dd6d91e4facc95) - base energy sample and quality metrics
- [`e5f01a1`](https://github.com/soran-ghaderi/torchebm/commit/e5f01a12f0aacf7a01cd0f6ad4c7f24a132a3e90) - base metric class

### Changed

- [`29f002d`](https://github.com/soran-ghaderi/torchebm/commit/29f002d528fed347ef8082bf0fc0ac93aff9817c) - rename param namings
- [`845f812`](https://github.com/soran-ghaderi/torchebm/commit/845f812ea74c0183626585a955bd95ad3aac80f6) - improved the perf of CD and the logic of PCD

### Fixed

- [`a22bceb`](https://github.com/soran-ghaderi/torchebm/commit/a22bceba119bcbb5e0bce498b4e9d8754d25f7c3) - commit changed files before gh-pages checkout

### Other

- [`1affb23`](https://github.com/soran-ghaderi/torchebm/commit/1affb23c41f823f8c23ae4ae479a490baa002366) - alter 'chain' to 'replay_buffer' in tests
- [`80739c4`](https://github.com/soran-ghaderi/torchebm/commit/80739c4f5f15912a6e68ac2727fb2ca15114b47b) - passing all tests
- [`b4bf745`](https://github.com/soran-ghaderi/torchebm/commit/b4bf7453ea08ed16707023383b632d2abf9b8863) - update README.md
- [`faee0a5`](https://github.com/soran-ghaderi/torchebm/commit/faee0a5a930078815b57f74ee08f80d6e47f511e) - fix code example - landing  page

## [0.3.0] - 2025-04-18

### Added

- [`cec7cfe`](https://github.com/soran-ghaderi/torchebm/commit/cec7cfe991f8f52c43d0d14557489f2c6c1511ba) - more data gen fns:
- [`59cbf35`](https://github.com/soran-ghaderi/torchebm/commit/59cbf3531a5f90d414e625e29d5c7ce950ce0259) - make_circle fn
- [`559ae9b`](https://github.com/soran-ghaderi/torchebm/commit/559ae9bb9813a6c0ae7057d916377bd8aa9032d2) - make_swiss_roll fn
- [`51830aa`](https://github.com/soran-ghaderi/torchebm/commit/51830aa0cd97851d5b1cb08e2af140bddc6b3126) - make_two_moons fn
- [`7591700`](https://github.com/soran-ghaderi/torchebm/commit/759170030535a4ecec4b298bffa7d04c3d162f82) - init datasets package

### Other

- [`f763ae8`](https://github.com/soran-ghaderi/torchebm/commit/f763ae896594bfda26a3c9c514638052989ab9ed) - dataset tests passing
- [`356308d`](https://github.com/soran-ghaderi/torchebm/commit/356308dc0feb04eef778aa286a70c0060ddfea5d) - fix landing page
- [`fe340d4`](https://github.com/soran-ghaderi/torchebm/commit/fe340d40fe473138f77ca4cf41deeec97dc51a42) - add assets
- [`e8aa0dc`](https://github.com/soran-ghaderi/torchebm/commit/e8aa0dc6320ee5e1aaae03704dc4ff41723c2e04) - update examples and the landing page
- [`c83e114`](https://github.com/soran-ghaderi/torchebm/commit/c83e11419a57bc49c0db3e969254a43d05c9b663) - contrastive div documentation
- [`d80ffb9`](https://github.com/soran-ghaderi/torchebm/commit/d80ffb98bb7134cf6dc89a5697db5bef229f71b5) - update docs

## [0.2.6] - 2025-04-13

### Added

- [`e74dba0`](https://github.com/soran-ghaderi/torchebm/commit/e74dba0ed18e22fd7879734753188ca6a660c3f5) - implement ContrastiveDivergence

### Changed

- [`77afe6b`](https://github.com/soran-ghaderi/torchebm/commit/77afe6b7b2bed6d0830e4aa845a16a98511ec25c) - renamed the core modules and classes

### Other

- [`57af9f3`](https://github.com/soran-ghaderi/torchebm/commit/57af9f36cb9c40a9f39bec23d79525cceac0a388) - base loss tests passing
- [`b5c649e`](https://github.com/soran-ghaderi/torchebm/commit/b5c649eda0fdb10e8ee6f8f00a2249ce037e6b32) - contrastive div tests passing
- [`f46371c`](https://github.com/soran-ghaderi/torchebm/commit/f46371c0ffb3b6b0aedf7d0fedba1b6450906e6d) - implement training an EBM
- [`417f101`](https://github.com/soran-ghaderi/torchebm/commit/417f101c99422ac5a74204abed5dae5f00b8c345) - fix img links in examples

## [0.2.4] - 2025-04-12

### Added

- [`0ed9ed7`](https://github.com/soran-ghaderi/torchebm/commit/0ed9ed7a3beb97d8a0b94a927e8945ed05545481) - HMC implemented
- [`d4a2ac1`](https://github.com/soran-ghaderi/torchebm/commit/d4a2ac11afb1190ba12997dd5c864188b7b52fde) - vectorized hmc
- [`b35323f`](https://github.com/soran-ghaderi/torchebm/commit/b35323fbb38d91ce21b93e6c8e661bfb2548252d) - return_trajectory and return_diagnostics support
- [`b12c3a4`](https://github.com/soran-ghaderi/torchebm/commit/b12c3a482914a597269befe8f8a0acd3adc29a88) - losses: ParallelTemperingCD draft
- [`51e96a6`](https://github.com/soran-ghaderi/torchebm/commit/51e96a64fe98aa4deb95365a6f845b1a849f1fb2) - losses: PersistentContrastiveDivergence draft
- [`474585b`](https://github.com/soran-ghaderi/torchebm/commit/474585b025be8aa82ec0ba73253283ff2b5b897f) - losses: ContrastiveDivergence
- [`60543ff`](https://github.com/soran-ghaderi/torchebm/commit/60543ff94cc898969ac7df76fdcee7308f6f08ab) - losses package
- [`4c5786c`](https://github.com/soran-ghaderi/torchebm/commit/4c5786c8655709d0f019efc713dc959def7cdafb) - Add multimodal energy and modified Langevin in examples
- [`463824b`](https://github.com/soran-ghaderi/torchebm/commit/463824b5e0bab1ff8e7ff36541cb4c0c10fabf4e) - implement HMC sampler and  visualization
- [`58df859`](https://github.com/soran-ghaderi/torchebm/commit/58df85906ddb630f919f0332ca7ff33b1974d792) - add energy function visualization and fix sum logic
- [`b37f4dd`](https://github.com/soran-ghaderi/torchebm/commit/b37f4dd1615aeb800fe116cd1a06ced7f243ac8e) - add RastriginEnergy class for energy calculations
- [`0abe25d`](https://github.com/soran-ghaderi/torchebm/commit/0abe25d75cf53efd4ce73b87dddca7c9f6b94758) - add new energy functions for optimization tasks
- [`af347c7`](https://github.com/soran-ghaderi/torchebm/commit/af347c761419ba55dafc38aa3ce18fa148224aa5) - add GaussianEnergy class for energy computation
- [`e75c00e`](https://github.com/soran-ghaderi/torchebm/commit/e75c00ea22c08c938387cf1a76de9b8453e4e3a1) - add DoubleWellEnergy class and improve device handling
- [`56e7b32`](https://github.com/soran-ghaderi/torchebm/commit/56e7b32500c69bb878b615a02fcf955067e8e8f8) - update langevin_dynamics.py
- [`26578b6`](https://github.com/soran-ghaderi/torchebm/commit/26578b656b2bd06e32fd2460e51c1c013f8b5f38) - update sampler with new features
- [`d58b8d0`](https://github.com/soran-ghaderi/torchebm/commit/d58b8d0cfbc2756a2ae5ce0b20965f9adcd40bbd) - initiate score_matching.py
- [`4797752`](https://github.com/soran-ghaderi/torchebm/commit/4797752f023b8e16174cef57a2e87a191386e06c) - initiate score_matching.py
- [`bd6635a`](https://github.com/soran-ghaderi/torchebm/commit/bd6635a6048748c7a941b8a762cde1ab036646f4) - LangevinDynamics
- [`8d8d310`](https://github.com/soran-ghaderi/torchebm/commit/8d8d31034a741f35f44a1a09627e91d2ef8085ef) - update ci/cd
- [`0d0bbf9`](https://github.com/soran-ghaderi/torchebm/commit/0d0bbf953e9fa1c0b39d6e6cf5a17bf374965513) - setup ci/cd
- [`fe166f7`](https://github.com/soran-ghaderi/torchebm/commit/fe166f7d795520bd95121f6a2e12ab6d58353b0e) - define main skeleton
- [`64e3e6e`](https://github.com/soran-ghaderi/torchebm/commit/64e3e6ec559f9944ac1c007fc3b1b042397c2ffb) - init commit

### Changed

- [`a4c86c4`](https://github.com/soran-ghaderi/torchebm/commit/a4c86c45faebc2d84e31f16eccf431870dcede5a) - #minor #release
- [`9fb30a5`](https://github.com/soran-ghaderi/torchebm/commit/9fb30a5e1ac8e3d35ecdace209db6b15606d7da0) - improved performance for langevin
- [`68886bd`](https://github.com/soran-ghaderi/torchebm/commit/68886bd6319fd7cbe63ee59e957457a29441bcaa) - comment out un-finalized codes
- [`e4214ca`](https://github.com/soran-ghaderi/torchebm/commit/e4214ca00d339e83f8c80577f60c4a177e38b1d9) - rename package, restructure examples, enhance samplers
- [`900c1b4`](https://github.com/soran-ghaderi/torchebm/commit/900c1b4ec7b4f908d89658c4c9af748600ac68c8) - improve energy functions and HMC sampler logic
- [`c8e893f`](https://github.com/soran-ghaderi/torchebm/commit/c8e893f68db3471b810014c3657b3ddf519e7141) - add images for the readme
- [`b7e5420`](https://github.com/soran-ghaderi/torchebm/commit/b7e5420436584a7b29467a337f9d5b527c701c6f) - LangevinDynamics
- [`26e87ac`](https://github.com/soran-ghaderi/torchebm/commit/26e87ace14f12760063a518f12b3c219bd4d97ed) - corrected branch for tag release action
- [`3838126`](https://github.com/soran-ghaderi/torchebm/commit/3838126c19eaa6eab91790c4fb236e9dff3d5c50) - update pyproject.toml and tag release action
- [`51b6839`](https://github.com/soran-ghaderi/torchebm/commit/51b683924fea985899b61069dd1e167aaf0eed60) - update pyproject.toml and tag release action #patch
- [`8c56eda`](https://github.com/soran-ghaderi/torchebm/commit/8c56edaf221b8b53eacf8da3e59b54c2f3a604f7) - update pyproject.toml and tag release action
- [`9fd16d6`](https://github.com/soran-ghaderi/torchebm/commit/9fd16d6da9f95c2ba114be64146cf8139115c7ae) - update pyproject.toml using actions 4
- [`e0fea5d`](https://github.com/soran-ghaderi/torchebm/commit/e0fea5db38c4c92823fd9abe56f2f3f53a9e5348) - update pyproject.toml using actions 3
- [`e0843a4`](https://github.com/soran-ghaderi/torchebm/commit/e0843a4874f7c79dc620c636115b042b31cb41e7) - update pyproject.toml using actions 2
- [`01fa1f6`](https://github.com/soran-ghaderi/torchebm/commit/01fa1f6f9f7613dbc502d2ffa1e20cc4ec701707) - update pyproject.toml using actions

### Fixed

- [`9f95ef7`](https://github.com/soran-ghaderi/torchebm/commit/9f95ef7a4aba02ec4e4ffec85883be122e25d09a) - fix pypi classifier
- [`cb5334d`](https://github.com/soran-ghaderi/torchebm/commit/cb5334d4b088e0231e2e7b3f869346e337171075) - fix the #release process
- [`03f7419`](https://github.com/soran-ghaderi/torchebm/commit/03f74192470683543c96b2dce884620b24f0e378) - mike paths for the api
- [`b19ef5a`](https://github.com/soran-ghaderi/torchebm/commit/b19ef5aab5ac9f28856f0cde52c2eba7346b5291) - calculating energy gradients
- [`377b364`](https://github.com/soran-ghaderi/torchebm/commit/377b3647c1903fcffb509597d2bec09963863006) - make it compatible with python 3.13
- [`564fd3b`](https://github.com/soran-ghaderi/torchebm/commit/564fd3ba7e686477104554506a45ae8092af5f0a) - add all potential libs to install
- [`2be8e57`](https://github.com/soran-ghaderi/torchebm/commit/2be8e572ac84c114729bf8ead5260c1bdefc4a53) - resolve python version discrepancy (docstrings)
- [`4024e89`](https://github.com/soran-ghaderi/torchebm/commit/4024e891cba856d2a465f3f5f600c32668993faa) - sample return trajectory fixed
- [`f8f7923`](https://github.com/soran-ghaderi/torchebm/commit/f8f79239484a0e80a030ae72d4797a582857dcb3) - ci/cd

### Other

- [`769a755`](https://github.com/soran-ghaderi/torchebm/commit/769a75525423d849282b416cf8b72d367e580cb2) - update pyproject.toml for #release
- [`7af48b6`](https://github.com/soran-ghaderi/torchebm/commit/7af48b6174b03d9a5e92df6d17218433eff933b0) - update pyproject.toml for #reslease
- [`3e12307`](https://github.com/soran-ghaderi/torchebm/commit/3e1230720fa49a1bfe3a8b84e2dac92c7de4926e) - skip tests if cuda not detected #release
- [`924d08f`](https://github.com/soran-ghaderi/torchebm/commit/924d08f97bf48b21d5565ade5028a3f014a21e89) - passing all tests locally
- [`d27dc94`](https://github.com/soran-ghaderi/torchebm/commit/d27dc94b6f7d7f13eee116f58cbe6521970aa497) - update examples and guides
- [`bc4ad10`](https://github.com/soran-ghaderi/torchebm/commit/bc4ad10094dbc3fadb6ebb5e9ca555591cf56dd2) - fixing plots
- [`0e4db76`](https://github.com/soran-ghaderi/torchebm/commit/0e4db768ce1a3cc11c2bec4dec4da6784d52d779) - #minor visualizations added
- [`0ddcdf7`](https://github.com/soran-ghaderi/torchebm/commit/0ddcdf7a342353c08a86c2706442d7f2776fa30d) - remove wrong req version from pyproject.toml
- [`9370202`](https://github.com/soran-ghaderi/torchebm/commit/93702028d30a677015551f7cabc76e355c9c039d) - remove wrong req version
- [`b43e235`](https://github.com/soran-ghaderi/torchebm/commit/b43e2359c43e53a7ff589f2d9561291fbee8fce7) - new website
- [`9c28304`](https://github.com/soran-ghaderi/torchebm/commit/9c28304e5c623c3d2cd14a350acf5d3069dd43fd) - update README.md with examples
- [`151640f`](https://github.com/soran-ghaderi/torchebm/commit/151640f4903f8d7334f615858537b8f574475818) - docs ci/cd
- [`54a14b2`](https://github.com/soran-ghaderi/torchebm/commit/54a14b29f3c55cbf2e6e10e1e275a69ed8a0b3e6) - docs ci/cd module links clickable
- [`9006bed`](https://github.com/soran-ghaderi/torchebm/commit/9006bed964c5ec9793bc01304ddacb9c6f59a886) - docs_ci remove the prev latest gh pages
- [`037c76d`](https://github.com/soran-ghaderi/torchebm/commit/037c76d6c43a2e1c60b48b47e1e3e83ff0c66fad) - docs_ci.yml remove --rebase
- [`0c8774a`](https://github.com/soran-ghaderi/torchebm/commit/0c8774a45dbac8addb1c8f12ce3c7ba239d6d45e) - docs_ci.yml deploying conflicts
- [`db07d76`](https://github.com/soran-ghaderi/torchebm/commit/db07d7636982ecaac5da762c530cc397be6ecb53) - passing all hmc tests
- [`f503a67`](https://github.com/soran-ghaderi/torchebm/commit/f503a67136980f8892833595a25f2203e04121bf) - updated the energy functions
- [`cbe0ece`](https://github.com/soran-ghaderi/torchebm/commit/cbe0eced13b9d0e7a7fd73fdb55f9becb5e4aa30) - all tests are passing
- [`23b67b7`](https://github.com/soran-ghaderi/torchebm/commit/23b67b7e6ec27c199a67c3a6439ea982aecc6be6) - hmc gaussian sampling
- [`3eb4035`](https://github.com/soran-ghaderi/torchebm/commit/3eb4035011b710992562bae12cdfb97ad94d4bb0) - init version of the HMC blog
- [`f3d8bfb`](https://github.com/soran-ghaderi/torchebm/commit/f3d8bfbe87be6b451e4531255a48fd503dde1996) - hmc blog
- [`7b5c2d8`](https://github.com/soran-ghaderi/torchebm/commit/7b5c2d812b8b2a42850dee68c1aa2385ed7e4d24) - debug github actions: revert toml file 1
- [`5d31149`](https://github.com/soran-ghaderi/torchebm/commit/5d31149d7242afb94a690825552efacd65ffa825) - debug github actions: revert toml file
- [`c2c8b3b`](https://github.com/soran-ghaderi/torchebm/commit/c2c8b3b84ec89619c46f352d0315729ddb5d864d) - printing 4
- [`bfaec11`](https://github.com/soran-ghaderi/torchebm/commit/bfaec11d1801783bf5ebbcc245da67e8160c36ca) - printing 3
- [`e09dfb4`](https://github.com/soran-ghaderi/torchebm/commit/e09dfb4625654800368a3aac2780c29dca5f3cec) - printing 2
- [`e00bd4e`](https://github.com/soran-ghaderi/torchebm/commit/e00bd4e9d8e5f2e749b0868a4a9f2034556faae0) - printing
- [`709d83a`](https://github.com/soran-ghaderi/torchebm/commit/709d83a150fd8b6891793aff816e7a65b3d48415) - updated ci.yml
- [`a7ef315`](https://github.com/soran-ghaderi/torchebm/commit/a7ef315466c30be5b8972bdc39058fe20156a2da) - updated the website structure
- [`be0905e`](https://github.com/soran-ghaderi/torchebm/commit/be0905eddd82d375f4bb8c03b45d4708d2f8bb00) - add docs for energy_function.py
- [`79acb3f`](https://github.com/soran-ghaderi/torchebm/commit/79acb3f7a38f64396f950bc2edbf66c85e9ba826) - remove redundant dependencies
- [`9792d0a`](https://github.com/soran-ghaderi/torchebm/commit/9792d0a30ea1f8da7dee55991c8ecfc690d8db40) - fix syntax err docs_ci.yml
- [`fb1e8f5`](https://github.com/soran-ghaderi/torchebm/commit/fb1e8f5983c8307f79dcd9c90b28631a4e440f23) - update docs_ci.yml -
- [`e96bc80`](https://github.com/soran-ghaderi/torchebm/commit/e96bc80cc28a7c5fa21935859e9ea84107f5697b) - update docs_ci.yml
- [`9e4f304`](https://github.com/soran-ghaderi/torchebm/commit/9e4f30418d1daffc95c9a932e427688aa251e9da) - update dev guide
- [`1843024`](https://github.com/soran-ghaderi/torchebm/commit/18430246169f6bf7db682e3acff9cd35a70f00c8) - remove everything related to poetry!
- [`2af1bff`](https://github.com/soran-ghaderi/torchebm/commit/2af1bff808364bb042b581c14190a6b99fcaf67a) - fix versioning with poetry - ci/cd
- [`2ea3cb6`](https://github.com/soran-ghaderi/torchebm/commit/2ea3cb66f1a4d548e5c47b13d512843554f50ccb) - fix setup and building the docs settings
- [`21f1165`](https://github.com/soran-ghaderi/torchebm/commit/21f11653db38f070e1c02598d28ea117a9939413) - initial API documentation works
- [`1edd278`](https://github.com/soran-ghaderi/torchebm/commit/1edd2789eb9bd3a81d58d03f869a3d1daed69b45) - update docs
- [`2003d9b`](https://github.com/soran-ghaderi/torchebm/commit/2003d9b6f5560ca3ddf5058484a27b195644cab2) - move lagevin_sampler_trajectory.py to examples
- [`3ace6a6`](https://github.com/soran-ghaderi/torchebm/commit/3ace6a6f5d1c56f3a0c5126535075dd0a705714d) - update references
- [`ccdea4d`](https://github.com/soran-ghaderi/torchebm/commit/ccdea4daaa181e6775e7f75d0497e313f3f4ab71) - add index.md
- [`3962bdd`](https://github.com/soran-ghaderi/torchebm/commit/3962bdda295e6f8115138b8630812c94cec7d3ac) - add mcdocs.yml
- [`5978704`](https://github.com/soran-ghaderi/torchebm/commit/59787048f40e0dad30a9aed08b5b05307d59c94f) - mcdoc test
- [`24f806e`](https://github.com/soran-ghaderi/torchebm/commit/24f806e6615ab53d4d85e34ef6ebf2054dc781b9) - documentation for the base sampler and langevin
- [`0dcdb07`](https://github.com/soran-ghaderi/torchebm/commit/0dcdb076d7b76d1d5808883535527f95ee4f931b) - langevin
- [`e23bf13`](https://github.com/soran-ghaderi/torchebm/commit/e23bf13520663436ec5795c71e5f526864d47d3e) - updated the langevin_dynamics_sampling.py
- [`ef5f4a1`](https://github.com/soran-ghaderi/torchebm/commit/ef5f4a1a24fba4632eec7e780888964ec8f7e210) - Refactor Langevin Dynamics sampling examples.
- [`3b54de4`](https://github.com/soran-ghaderi/torchebm/commit/3b54de41580dc3447eaae01a9a7508d9de9def04) - examples updated
- [`9d844b0`](https://github.com/soran-ghaderi/torchebm/commit/9d844b0005f58454f058dd7913c6d657c169ef64) - Create README.md
- [`3b69259`](https://github.com/soran-ghaderi/torchebm/commit/3b69259a1abd9639ac42fbb142c056b1a31adba2) - langevin #release
- [`9264660`](https://github.com/soran-ghaderi/torchebm/commit/92646603e1c7ee9452f8ecc1573b0c3fe385d5aa) - langevin - passes input parasm, sample()
- [`730f00e`](https://github.com/soran-ghaderi/torchebm/commit/730f00e8baa8c17b7a02c8a11f194685eb181b89) - refactor and langevin_dynamics initialization test
- [`abcd180`](https://github.com/soran-ghaderi/torchebm/commit/abcd1809b7dc46c3401ebbfee7cbd3efe28ec741) - update test for ci/cd 3
- [`084bbdf`](https://github.com/soran-ghaderi/torchebm/commit/084bbdf5c01e52bce95c74b974ef83cf94f7de11) - update test for ci/cd
- [`d468573`](https://github.com/soran-ghaderi/torchebm/commit/d4685734a43f8e286961f6297e50f5e034eecc04) - Update tag-release.yml
- [`0603a18`](https://github.com/soran-ghaderi/torchebm/commit/0603a1862ec8bbbff241f2b15033585d3fa8e492) - update test for ci/cd
- [`6007e01`](https://github.com/soran-ghaderi/torchebm/commit/6007e017ecc02c2d508cb4492fb80c74fecfd097) - example test for ci/cd
- [`12f1490`](https://github.com/soran-ghaderi/torchebm/commit/12f149092e5d034390b551cdcf388fb3a885498d) - add tests

[Unreleased]: https://github.com/soran-ghaderi/torchebm/compare/v0.7.5...HEAD
[0.7.5]: https://github.com/soran-ghaderi/torchebm/compare/v0.7.1...v0.7.5
[0.7.1]: https://github.com/soran-ghaderi/torchebm/compare/v0.6.0...v0.7.1
[0.6.0]: https://github.com/soran-ghaderi/torchebm/compare/v0.5.8...v0.6.0
[0.5.8]: https://github.com/soran-ghaderi/torchebm/compare/v0.5.7...v0.5.8
[0.5.7]: https://github.com/soran-ghaderi/torchebm/compare/v0.5.6...v0.5.7
[0.5.6]: https://github.com/soran-ghaderi/torchebm/compare/v0.5.4...v0.5.6
[0.5.4]: https://github.com/soran-ghaderi/torchebm/compare/v0.5.1...v0.5.4
[0.5.1]: https://github.com/soran-ghaderi/torchebm/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/soran-ghaderi/torchebm/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/soran-ghaderi/torchebm/compare/v0.3.17...v0.4.0
[0.3.17]: https://github.com/soran-ghaderi/torchebm/compare/v0.3.7...v0.3.17
[0.3.7]: https://github.com/soran-ghaderi/torchebm/compare/v0.3.5...v0.3.7
[0.3.5]: https://github.com/soran-ghaderi/torchebm/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/soran-ghaderi/torchebm/compare/v0.3.0...v0.3.4
[0.3.0]: https://github.com/soran-ghaderi/torchebm/compare/v0.2.6...v0.3.0
[0.2.6]: https://github.com/soran-ghaderi/torchebm/compare/v0.2.4...v0.2.6
[0.2.4]: https://github.com/soran-ghaderi/torchebm/releases/tag/v0.2.4
