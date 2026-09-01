r"""Deprecation machinery: self-registering metadata, warnings, removal windows.

A deprecation is declared where it lives, via `deprecated` (classes and
functions) or a module-level `declare_deprecation` (in-function paths).
Declaring registers a `DeprecationInfo` in `DEPRECATIONS` at import time;
``tests/test_deprecations.py`` imports every package module, cross-checks
each declaration site against this registry, and fails the suite once a
removal window closes. No removal version is ever written down.

Window rule (the policy's single home): a window closes only when BOTH
conditions hold, releases and calendar time.

- Releases: ``since`` is the first release that ships the warning; stamp
  the upcoming release when declaring (if another release lands first,
  re-stamp on merge). The window stays open for ``grace`` releases starting
  at ``since`` (the release automation bumps one patch per release) and
  version-expires after that, or at any newer minor/major, whichever comes
  first.
- Time: ``deprecated_on`` is the declaration date; the window never closes
  earlier than ``GRACE_MONTHS`` calendar months after it, so a burst of
  releases cannot strip users of migration time.

Example: ``since="0.8.3"``, ``grace=2``, ``deprecated_on="2026-08-31"``
warns in 0.8.3 and 0.8.4; removal is due at 0.8.5 (or 0.9.0), but not
before 2026-10-31. The tests gate one release ahead under patch cadence; a
deliberate minor/major release closes windows at its own tag, so run the
removals before jumping.

This module must stay importable without torch (stdlib only): it runs at
import time of every module that declares a deprecation.
"""

from __future__ import annotations

import calendar
import datetime
import functools
import os
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

GRACE = 2
GRACE_MONTHS = 2

_RELEASE_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


class TorchEBMDeprecationWarning(DeprecationWarning):
    r"""Deprecation warning emitted by torchebm.

    Subclasses `DeprecationWarning`, so generic filters and
    ``pytest.warns(DeprecationWarning)`` keep matching, while
    torchebm-specific filtering stays possible.
    """


#: Once-per-process warning keys. Tests clear this set to force re-warns.
_WARNED_ONCE: set = set()


def warn_once(
    key: str,
    message: str,
    category: type = DeprecationWarning,
    stacklevel: int = 3,
) -> None:
    r"""Emit a warning at most once per process, keyed by `key`.

    Deprecation paths on hot code (per-step sampler loops, per-batch losses)
    must not call `warnings.warn` every iteration: even when the filter shows
    a warning only once, the per-call filter processing adds avoidable
    overhead. This guard skips the call entirely after the first hit.
    """
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    warnings.warn(message, category, stacklevel=stacklevel)


def _parse_release(version: str) -> Tuple[int, int, int]:
    r"""Leading ``major.minor.patch`` of a PEP 440 version string.

    An epoch prefix is stripped; dev/rc/post/local suffixes are ignored, so a
    pre-release of X gates exactly like X (the gate fires during the rc/dev
    cycle of the closing release, which is the desired lead time).

    Raises:
        ValueError: If no three-component release prefix is present.
    """
    v = re.sub(r"^\d+!", "", version.strip())
    m = _RELEASE_RE.match(v)
    if m is None:
        raise ValueError(f"cannot parse a major.minor.patch release from {version!r}")
    return (int(m[1]), int(m[2]), int(m[3]))


def _add_months(day: datetime.date, months: int) -> datetime.date:
    month_index = day.month - 1 + months
    year = day.year + month_index // 12
    month = month_index % 12 + 1
    return datetime.date(
        year, month, min(day.day, calendar.monthrange(year, month)[1])
    )


def _find_stacklevel() -> int:
    r"""Stacklevel that attributes a warning to the first caller outside
    torchebm and torch (the pandas ``find_stack_level`` idiom)."""
    torch_mod = sys.modules.get("torch")
    torch_dir = (
        os.path.dirname(os.path.abspath(torch_mod.__file__))
        if torch_mod is not None and getattr(torch_mod, "__file__", None)
        else None
    )
    frame = sys._getframe()
    level = 0
    while frame is not None:
        filename = frame.f_code.co_filename
        internal = filename.startswith(_PACKAGE_DIR + os.sep) or (
            torch_dir is not None and filename.startswith(torch_dir + os.sep)
        )
        if level > 0 and not internal:
            break
        frame = frame.f_back
        level += 1
    # warnings.warn runs inside warn_once, one frame deeper than this
    # computation, so the target frame sits one level further out.
    return level + 1


@dataclass(frozen=True)
class DeprecationInfo:
    r"""One registered deprecation. Construct via `declare_deprecation`."""

    name: str
    since: str
    deprecated_on: str
    replacement: str
    module: str
    grace: int = GRACE
    grace_months: int = GRACE_MONTHS
    message: str = ""
    removal: str = ""

    @property
    def key(self) -> Tuple[str, str]:
        return (self.module, self.name)

    def _full_message(self) -> str:
        return self.message or (
            f"{self.name} is deprecated; use {self.replacement} instead."
        )

    def warn(self, qualifier: str = "") -> None:
        r"""Emit the once-per-process warning for this deprecation.

        ``qualifier`` extends the dedup key so one declaration can warn once
        per concrete API (e.g. once per loss class). The warning is
        attributed to the first stack frame outside torchebm/torch.

        Raises:
            RuntimeError: If this info was never registered; construct it
                through `declare_deprecation`, not directly.
        """
        if DEPRECATIONS.get(self.key) != self:
            raise RuntimeError(
                f"deprecation {self.key} is not registered; declare it with "
                "declare_deprecation() at module level"
            )
        dedup_key = f"{self.module}:{self.name}:{qualifier}"
        if dedup_key in _WARNED_ONCE:
            return
        warn_once(
            dedup_key,
            self._full_message(),
            category=TorchEBMDeprecationWarning,
            stacklevel=_find_stacklevel(),
        )

    def removal_due(
        self, current_version: str, today: Optional[datetime.date] = None
    ) -> bool:
        r"""Whether the window is past for ``current_version`` on ``today``.

        Both conditions of the module-docstring rule must hold: the release
        window is version-expired AND at least ``grace_months`` calendar
        months have passed since ``deprecated_on``. A current release below
        ``since`` is never due: stale dev installs lag the released anchors.
        """
        cur = _parse_release(current_version)
        anchor = _parse_release(self.since)
        if cur[:2] != anchor[:2]:
            version_due = cur[:2] > anchor[:2]
        else:
            version_due = cur[2] - anchor[2] >= self.grace
        if not version_due:
            return False
        if today is None:
            today = datetime.date.today()
        floor = _add_months(
            datetime.date.fromisoformat(self.deprecated_on), self.grace_months
        )
        return today >= floor


DEPRECATIONS: Dict[Tuple[str, str], DeprecationInfo] = {}


def _register(info: DeprecationInfo) -> DeprecationInfo:
    existing = DEPRECATIONS.get(info.key)
    if existing is not None and existing != info:
        raise ValueError(
            f"conflicting declarations for deprecation {info.key}: "
            f"{existing} vs {info}"
        )
    DEPRECATIONS[info.key] = info
    return info


def declare_deprecation(
    *,
    name: str,
    since: str,
    deprecated_on: str,
    replacement: str,
    module: Optional[str] = None,
    grace: int = GRACE,
    grace_months: int = GRACE_MONTHS,
    message: str = "",
    removal: str = "",
) -> DeprecationInfo:
    r"""Declare a deprecated in-function path; call at module level.

    Module-level declaration makes registration an import-time effect, so
    the ledger tests see the entry without the deprecated path ever running.
    Re-declaring identically (module reload) is a no-op; re-declaring with
    different fields raises.

    Args:
        name: The deprecated surface, as a user would name it.
        since: First release shipping the warning (validated eagerly).
        deprecated_on: ISO declaration date; the window never closes before
            ``grace_months`` months after it (validated eagerly).
        replacement: What users migrate to.
        module: Declaring module; defaults to the caller's ``__name__``.
            Pass ``module=__name__`` explicitly when wrapping this call.
        grace: Releases the window stays open, counting ``since``.
        grace_months: Calendar months the window stays open at minimum.
        message: Full warning text; derived from name/replacement if empty.
        removal: What to delete when the window closes (shown by the gate).
    """
    if module is None:
        module = sys._getframe(1).f_globals.get("__name__", "<unknown>")
    _parse_release(since)
    datetime.date.fromisoformat(deprecated_on)
    return _register(
        DeprecationInfo(
            name=name,
            since=since,
            deprecated_on=deprecated_on,
            replacement=replacement,
            module=module,
            grace=int(grace),
            grace_months=int(grace_months),
            message=message,
            removal=removal,
        )
    )


def deprecated(
    *,
    since: str,
    deprecated_on: str,
    replacement: str,
    grace: int = GRACE,
    grace_months: int = GRACE_MONTHS,
    message: str = "",
    removal: str = "",
) -> Callable:
    r"""Decorator deprecating a class or function.

    Registers on import and warns on every use with a plain
    ``warnings.warn`` (call sites are not hot paths; the default
    once-per-location filter handles repetition). A decorated class keeps
    its identity: ``__init__`` is wrapped in place, so isinstance, pickling,
    deepcopy, subclassing, and signature introspection are unaffected; a
    subclass reaching the wrapped ``__init__`` warns too, which is correct
    for a deprecated base. Sets the PEP 702 ``__deprecated__`` attribute so
    type checkers and IDEs can flag use sites.
    """

    def apply(obj):
        info = declare_deprecation(
            name=obj.__qualname__,
            since=since,
            deprecated_on=deprecated_on,
            replacement=replacement,
            module=obj.__module__,
            grace=grace,
            grace_months=grace_months,
            message=message,
            removal=removal,
        )
        msg = info._full_message()
        if isinstance(obj, type):
            wrapped_init = obj.__init__

            @functools.wraps(wrapped_init)
            def __init__(self, *args, **kwargs):
                warnings.warn(msg, TorchEBMDeprecationWarning, stacklevel=2)
                wrapped_init(self, *args, **kwargs)

            obj.__init__ = __init__
            obj.__deprecated__ = msg
            return obj

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, TorchEBMDeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        wrapper.__deprecated__ = msg
        return wrapper

    return apply
