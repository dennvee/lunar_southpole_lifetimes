import argparse
import csv
import shutil
from pathlib import Path
from time import sleep
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cratermaker import Simulation
from cratermaker.bindings import basicmoon_bindings
from cratermaker.components.morphology.basicmoon import BasicMoonCrater
from cratermaker.utils.general_utils import format_large_units
from matplotlib import rcParams
from pyproj import Geod
from scipy.optimize import curve_fit
from shapely import difference, make_valid
from shapely.geometry import Polygon
from tqdm import tqdm
from xarray.groupers import BinGrouper

scriptdir = Path(__file__).resolve().parent
rootdir = scriptdir.parent
datadir = rootdir / "data"
simrootdir = rootdir / "simdata"
plotdir = rootdir / "plots"
default_input_csv = datadir / "fresh_craters.csv"
vectorfiledir = datadir / "vector_profiles"
classification_file = datadir / "crater_classification.csv"
resolution_file = datadir / "crater_resolution.csv"

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

_REGION_SIZE_FACTOR = 3.00

valid_fit_vars = [
    "radius",
    "floor_elevation",
    "floor_radius",
    "wall_curvature",
    "floor_blend",
    "rim_width",
    "rim_height",
    "frac_ejrim",
    "ejprofile",
    "peak_height",
    "peak_width",
    "peak_ring_radius",
    "peak_center_distance",
    "peak_center_bearing",
]

notused_vars = ["peak_center_distance", "peak_center_bearing"]

default_fit_vars = [v for v in valid_fit_vars if v not in notused_vars]

ring_linked_vars = [
    "floor_elevation",
    "peak_height",
    "peak_width",
    "peak_ring_radius",
    "floor_blend",
    "location",
]


def ring_prefix(i):
    """
    Returns a string that is prepended to variables representing the ith ring associated with a crater.
    """
    return f"ring{i + 1:02d}"


dummysim = Simulation(
    simdir=simrootdir / "dummy",
    gridlevel=0,
    morphology="basicmoon",
    ask_overwrite=False,
    reset=False,
)


class DummyCrater:
    """
    A lightweight class that is used to pass morphometric parameters to the Rust profile function during fitting without the overhead associated with a full Cratermaker Crater object.

    Notes: This class defines parameters using a unit system relative to radius
    """

    def __init__(
        self,
        popt_map: dict,
        psig_map: dict,
        crater: BasicMoonCrater | None = None,
        Crater: type[BasicMoonCrater] = dummysim.Crater,
        **kwargs: Any,
    ):
        self.rings = None
        if crater is None:
            crater = Crater.maker(**{**popt_map, **kwargs})
        ring_popt_maps, ring_psig_maps = None, None
        _popt_map = popt_map.copy()
        _psig_map = psig_map.copy()
        param_list = [
            "name",
            "label",
            "id",
            "parent",
            "morphology_type",
            "location",
            "dem_file_list",
            "res",
        ]
        if "D/pix" in kwargs:
            v = kwargs.pop("D/pix")
            kwargs["res"] = v
        location = (
            float(kwargs.pop("longitude", crater.location[0])),
            float(kwargs.pop("latitude", crater.location[1])),
        )
        if "location" not in kwargs:
            kwargs["location"] = location
        if "dem_file_list" in kwargs:
            dflstr = kwargs.pop("dem_file_list").item()
            # Here we do some complicated character stuff in order to convert strings that look like this:
            #    "['url']""
            # or
            #    "['url1';'url2']""
            #  into lists. We substitute various characters and use "<" as a temporarily deliminter, as it is not a valid url character
            if isinstance(dflstr, str):
                dflstr = (
                    dflstr.replace(" ", "")
                    .replace('"', "'")
                    .replace("[", "<")
                    .replace("]", "<")
                    .replace(",", ";")
                    .split("<")
                )
                if len(dflstr) == 3:
                    dflstr = dflstr[1]
                else:
                    dflstr = dflstr[0]
                dflstr = dflstr.replace("';'", "<").replace("'", "").split("<")
                kwargs["dem_file_list"] = dflstr
        else:
            kwargs["dem_file_list"] = None

        if "diameter" in kwargs:
            kwargs["radius"] = 0.5 * kwargs.pop("diameter")
        keys = list(kwargs.keys())

        popt_direct_set = []
        if len(_popt_map) > 0:
            for k, v in _popt_map.items():
                if k in valid_fit_vars:
                    object.__setattr__(self, k, v)
                    popt_direct_set.append(k)
            if len(popt_direct_set) > 0:
                for k in popt_direct_set:
                    _popt_map.pop(k)

        psig_direct_set = []
        if len(_psig_map) > 0:
            for k, v in _psig_map.items():
                if k in valid_fit_vars:
                    k_std = f"{k}_std"
                    object.__setattr__(self, k_std, v)
                    psig_direct_set.append(k)
            if len(psig_direct_set) > 0:
                for k in psig_direct_set:
                    _psig_map.pop(k)

        if len(keys) > 0:
            for k in keys:
                if any(varname in k for varname in valid_fit_vars):
                    v = kwargs.pop(k)
                    if "_std" in k:
                        _psig_map[f"{k.replace('_std', '')}"] = v
                    else:
                        _popt_map[k] = v

        if len(_popt_map) > 0:
            fit_vars, popt = zip(*_popt_map.items())
            if len(_psig_map) > 0:
                keys = list(_psig_map.keys())
                for k in keys:
                    if k not in _popt_map:
                        _psig_map.pop(k)
                for k in _popt_map:
                    if k not in _psig_map:
                        _psig_map[k] = 0.0
                _, psig = zip(*_psig_map.items())
            else:
                psig = None

            _popt_map, _psig_map, ring_popt_maps, ring_psig_maps = get_ring_popt_maps(
                fit_vars, popt, psig
            )
        else:
            _popt_map, _psig_map, ring_popt_maps, ring_psig_maps = {}, {}, None, None

        rings = None
        if crater.nrings > 0:
            rings = []
            for ring in crater.rings:
                rings.append(self.__class__(crater=ring, popt_map={}, psig_map={}))

        if ring_popt_maps is not None:
            if rings is None:
                rings = [crater] * len(ring_popt_maps)
            for i, ring_popt_map in enumerate(ring_popt_maps):
                if len(rings) > i:
                    ring = rings[i]
                    if ring_psig_maps is not None:
                        ring_psig_map = ring_psig_maps[i]
                    else:
                        ring_psig_map = {}
                    rings[i] = self.__class__(
                        crater=ring, popt_map=ring_popt_map, psig_map=ring_psig_map
                    )

        self.rings = rings

        for k in param_list:
            if k in kwargs:
                v = kwargs[k]
            elif hasattr(crater, k):
                v = getattr(crater, k)
            else:
                v = None
            if isinstance(v, np.ndarray):
                v = v.item()
            object.__setattr__(self, k, v)

        for k in valid_fit_vars:
            if k not in popt_direct_set:
                if _popt_map is not None and k in _popt_map:
                    v = _popt_map.pop(k)
                elif hasattr(crater, k):
                    v = getattr(crater, k)
                else:
                    v = None
                if isinstance(v, np.ndarray):
                    v = v.item()
                if v is not None:
                    object.__setattr__(self, k, v)

            if k not in psig_direct_set:
                k_std = f"{k}_std"
                if _psig_map is not None and k in _psig_map:
                    v_std = _psig_map.pop(k)
                elif hasattr(crater, k_std):
                    v_std = getattr(crater, k_std)
                else:
                    v_std = None
                if isinstance(v_std, np.ndarray):
                    v_std = v_std.item()
                if v_std is not None and np.isfinite(v_std):
                    object.__setattr__(self, k_std, v_std)

        # Compute dependent parameters and adjust any values that are out of bounds
        self.diameter = self.radius * 2
        if not isinstance(self.name, str) and isinstance(self.label, str):
            if "F_" in self.label:
                self.name = self.label.replace("F_", "")
            elif "CS_" in self.label:
                self.name = self.label.replace("CS_", "")
            else:
                self.name = self.label
            if self.name != self.label:
                if "_" in self.name:
                    self.name = self.name.split("_")[0]
                if self.diameter > 1000:
                    self.name = f"D{self.diameter * 1e-3:.1f}km in {self.name}"
                else:
                    self.name = f"D{self.diameter:.1f}m in {self.name}"
            else:
                if "-" in self.name:
                    self.name = f"Robbins ID {self.name}"
                else:
                    self.name = "".join(
                        " " + c if c.isupper() else c.strip() for c in self.name
                    ).strip()

        if hasattr(self, "radius_std"):
            self.diameter_std = 2 * self.radius_std
        self.floor_radius = min(max(self.floor_radius, 0.0), 0.9 * self.radius)
        self.floor_elevation = min(max(self.floor_elevation, -self.radius), 0.0)
        self.wall_curvature = min(max(self.wall_curvature, 0.0), 1.0)
        self.floor_blend = min(max(self.floor_blend, 0.0), 1.0)
        self.rim_width = min(max(self.rim_width, 1e-2 * self.radius), 0.5 * self.radius)
        self.ejprofile = min(max(self.ejprofile, -6.0), -0.25)
        self.frac_ejrim = min(max(self.frac_ejrim, 0.0), 1.0)
        self.peak_height = min(max(self.peak_height, 0.0), 2 * self.floor_radius)
        self.peak_width = min(
            max(self.peak_width, self.peak_height), 2 * self.floor_radius
        )
        self.peak_ring_radius = min(
            max(self.peak_ring_radius, 0.0), 2 * (self.floor_radius + self.peak_width)
        )
        self.depth = self.rim_height - self.floor_elevation
        self.depth_to_diameter = self.depth / self.diameter
        if hasattr(self, "rim_height_std") and hasattr(self, "floor_elevation_std"):
            self.depth_std = self.rim_height_std + self.floor_elevation_std
            if hasattr(self, "diameter_std"):
                self.depth_to_diameter_std = self.depth * np.sqrt(
                    (self.depth_std / self.depth) ** 2
                    + (self.diameter_std / self.diameter) ** 2
                )
        if self.morphology_type != "ring":
            self.parent = None
        if not hasattr(self, "dem_file_list"):
            self.dem_file_list = None
        if self.rings is not None:
            for i in range(len(self.rings)):
                self.rings[i].morphology_type = "ring"
                self.rings[i].parent = self.id
                self.rings[i].radius = min(self.rings[i].radius, self.radius)
                self.rings[i].diameter = self.rings[i].radius * 2
                for v in ring_linked_vars:
                    setattr(self.rings[i], v, getattr(self, v))
                    v_std = f"{v}_std"
                    if hasattr(self, v_std):
                        setattr(self.rings[i], v_std, getattr(self, v_std))

                self.rings[i].frac_ejrim = 0.0
                self.rings[i].ejprofile = -3.0
                if hasattr(self, "frac_ejrim_std"):
                    self.rings[i].frac_ejrim_std = self.frac_ejrim_std

            self.rings.sort(key=lambda ring: -ring.radius)
            for i in range(len(self.rings)):
                if self.id < 10000:
                    self.rings[i].id = self.id * 100 + (i + 1)
                else:
                    self.rings[i].id = self.id + i + 1

    @property
    def as_popt_map(self):
        """
        Returns crater attribues as a popt_map where keys are the fit parameters and values are the attribute values, including ring parameters in the form of "ring0N_parameter", where N is the ring number starting with 1.
        """

        popt = {}
        for k in valid_fit_vars:
            if hasattr(self, k) and k not in notused_vars:
                popt[k] = getattr(self, k)

        if self.nrings > 0:
            for i, ring in enumerate(self.rings):
                pre = ring_prefix(i)
                for k in valid_fit_vars:
                    # Remove variables that are linked to the main crater object
                    if k in ring_linked_vars or k in notused_vars:
                        continue
                    popt[f"{pre}_{k}"] = getattr(ring, k)
        return popt

    @property
    def as_psig_map(self):
        """
        Returns crater attribues as a popt_map where keys are the fit parameters and values are the attribute values, including ring parameters in the form of "ring0N_parameter", where N is the ring number starting with 1.
        """

        psig = {}
        for k in valid_fit_vars:
            k_std = f"{k}_std"
            if hasattr(self, k_std) and k not in notused_vars:
                psig[k] = getattr(self, k_std)

        if self.nrings > 0:
            for i, ring in enumerate(self.rings):
                pre = ring_prefix(i)
                for k in valid_fit_vars:
                    # Remove variables that are linked to the main crater object
                    if k in ring_linked_vars or k in notused_vars:
                        continue
                    k_std = f"{k}_std"
                    if hasattr(ring, k_std):
                        psig[f"{pre}_{k}"] = getattr(ring, k_std)
        return psig

    @property
    def nrings(self):
        """Returns the total number of rings attached to the object."""
        if not hasattr(self, "rings") or self.rings is None:
            return 0
        else:
            return len(self.rings)

    def ring_offset(self, ringnum):
        return (
            profile_function(
                self,
                None,
                np.array([self.rings[ringnum].radius]),
                None,
                True,
                basicmoon_bindings.basicmoon_profile,
                self.__class__,
            )[0]
            - self.rings[ringnum].rim_height
        )

    def to_cratermaker(self, sim):
        main_args = self.as_popt_map
        crater = sim.Crater.maker(
            name=self.name,
            location=self.location,
            id=self.id,
            morphology_type=self.morphology_type,
            make_ring=False,
            **main_args,
        )
        if self.nrings > 0:
            crater._rings = []
            for ring in self.rings:
                ring_args = ring.as_popt_map
                crater.add_ring(**ring_args)
        return crater

    def as_dict(self, skip_complex_data: bool = False, **kwargs):
        d = self.__dict__.copy()
        if skip_complex_data:
            d.pop("rings", None)
        return d


def get_input_craters(
    infile: Path | str, Crater: type[BasicMoonCrater] = dummysim.Crater
):
    """
    Parses a CSV file and builds a

    Parameters
    ----------
    infile : Path
        The path to the input CSV

    Returns
    -------
    list[DummyCrater]
        A list of DummyCrater objects retrieved from the input file.
    """
    ds = pd.read_csv(infile).to_xarray()
    if "radius" in ds:
        sizefield = "radius"
    elif "diameter" in ds:
        sizefield = "diameter"
    ds = ds.sortby(sizefield)
    craters = Crater.from_xarray(ds)
    ds = ds.set_coords(["id"])
    cds = pd.read_csv(classification_file).to_xarray().set_coords(["label"])
    resds = pd.read_csv(resolution_file).to_xarray().set_coords(["label"])
    resds = resds.where(resds["morphology_type"] != "ring", drop=True)
    for i, crater in enumerate(craters):
        ids = ds.sel(id=crater.id)
        kwargs = {
            k: v["data"] for k, v in ids.to_dict(data="array")["data_vars"].items()
        }
        if crater.nrings > 0:
            ring_ids = (
                ds.where(ds.parent == crater.id, drop=True)
                .set_coords([sizefield])
                .sortby(sizefield, ascending=False)
                .id.data
            )

            for j, rid in enumerate(ring_ids):
                jds = ds.sel(id=rid)
                kwargs = {
                    **kwargs,
                    **{
                        f"ring{j + 1:02d}_{k}": v["data"]
                        for k, v in jds.to_dict(data="array")["data_vars"].items()
                    },
                }
        crater = DummyCrater(crater=crater, popt_map={}, psig_map={}, **kwargs)
        if crater.label in cds.label:
            crater.classification = cds.sel(
                label=crater.label
            ).classification.data.item()
        else:
            crater.classification = None
        if crater.label in resds.label:
            crater.res = resds.sel(label=crater.label)["D/pix"].item()
        else:
            crater.res = 1e6

        craters[i] = crater

    return craters


def make_complex_ring(crater: BasicMoonCrater | DummyCrater, nrings: int):
    """
    Creates a set of rings that can be used as an initial guess for the fitting function if none were provided in the file.

    Parameters
    ----------
    crater : BasicMoonCrater | DummyCrater
        The original Crater object to add rings to.
    nrings : int
        The number of rings to generate.
    """
    crater = DummyCrater(crater=crater, popt_map={}, psig_map={})
    crater.rings = []
    ring = crater
    for i in range(nrings):
        ring = DummyCrater(
            crater=ring,
            radius=crater.radius / np.sqrt(2.0) ** (i + 1),
            frac_ejrim=0.0,
            ejprofile=-3,
            floor_radius=ring.floor_radius * 0.75,
            rim_width=0.3 * ring.rim_width,
            rim_height=0.1 * ring.rim_height,
            popt_map={},
            psig_map={},
        )
        crater.rings.append(ring)
    crater = DummyCrater(crater=crater, popt_map={}, psig_map={})

    return crater


def get_morphometry_from_binned_profile(
    sim: Simulation,
    crater: BasicMoonCrater | DummyCrater,
    fit_vars: list[str, ...] = default_fit_vars,
    nrbins: int = 500,
    fit_range: tuple = (None, None),
    plot_extension: str = "png",
    save_intermediate: bool = False,
    fit_rim: bool = True,
    plotfile: str | Path | None = None,
):
    """
    Fit crater morphology parameters to a binned radial elevation profile.

    Parameters
    ----------
    sim : cratermaker.Simulation
        Active simulation object used to construct crater profiles.
    crater : cratermaker.Crater
        Crater object supplying initial morphology estimates.
    fit_vars : list[str,...]
        List of morphometric variables to fit.
    nrbins : int
        Number of radial bins used to define the binned profile.
    fit_range : tuple
        A tuple that gives the range of radius points that will be used for the fit relative to the crater radius. Examples are
        (None, None) : use the full range.
        (None, 1.25) : use all points inward of 1.25 * crater.radius.
        (0.5, 1.5) : use points between 0.5 * crater.radius to 1.5 * crater.radius.
    plot_extension : str, optional
        File extension for the profile plot (e.g., "pdf", "png"). Default is "png".
    save_intermediate : bool, optional
        Whether or not to plot the original profile before the fit was performed. Default is False
    fit_rim : bool, optional
        If False, but "radius" appears in the fit_vars, then radius will be determined as the radial bin with the highest elevation rather than be fit.

    Returns
    -------
    popt_map : dict
        A dict of fit variables as keys and the best-fit parameters from ``scipy.optimize.curve_fit`` as values
    psig_map : dict
        A dict of fit variables as keys and the standard error (sqrt of diagonal of covariance matrix of fitted parameters from ``scipy.optimize.curve_fit``) as values.

    Notes
    -----
    A plot showing the data and the profile fit is written to ``plotdir / f"{crater.label}_profile.{plot_extension}"``.
    """
    crater = DummyCrater(crater=crater, popt_map={}, psig_map={})
    fit_vars_orig = fit_vars.copy()
    region = get_region(sim, crater, fit_range)
    ivals = np.argsort(region.face_distance)
    rvals = region.face_distance[ivals]
    hvals = region.desloped_face_elevation[ivals]
    ymin, ymax = 1.5 * min(hvals), 1.5 * max(hvals)
    rlo = fit_range[0] * crater.radius if fit_range[0] is not None else min(rvals)
    rhi = fit_range[1] * crater.radius if fit_range[1] is not None else max(rvals)
    inrange = (rvals >= rlo) & (rvals <= rhi)

    rbin, hbin, hstd = binned_profile(rvals[inrange], hvals[inrange], nrbins=nrbins)

    if "floor_elevation" not in fit_vars:
        # Use the binned profile to get the floor elevation value and use this for all slices
        ifloor = np.argmin(hbin)
        crater = DummyCrater(
            crater=crater,
            popt_map={"floor_elevation": hbin[ifloor]},
            psig_map={"floor_elevation": hstd[ifloor]},
        )

    # Check to make sure that crater parameters aren't widely differnt than what the profile gives (this can happen when rings end up acting as floors)
    hmin = np.min(hbin)
    ntries = 10
    for i in range(10):
        if crater.floor_elevation < 1.1 * hmin:
            crater = DummyCrater(
                crater=crater, popt_map={}, psig_map={}, floor_elevation=hmin
            )
        if "radius" in fit_vars and not fit_rim:
            radius, rim_height = get_rim_from_data(rbin, hbin, crater)
            crater = DummyCrater(
                crater=crater,
                radius=radius,
                rim_height=rim_height,
                popt_map={},
                psig_map={},
            )

        fit_bounds = {"floor_elevation": (1.1 * hmin, 0)}
        horig = profile_function(
            crater,
            fit_vars,
            rbin,
            None,
            True,
            basicmoon_bindings.basicmoon_profile,
            DummyCrater,
        )
        fv = [v for v in fit_vars_orig]
        popt_map, psig_map, intermediate_results = fit_profile(
            sim,
            crater,
            fv,
            rbin,
            hbin,
            fit_bounds=fit_bounds,
            fit_rings=True,
            fit_rim=fit_rim,
            save_intermediate=save_intermediate,
        )
        intermediate_results = {
            "Initial guess": {
                "popt_map": crater.as_popt_map,
                "psig_map": crater.as_psig_map,
                "morphometric_vars": fv,
                "plot_xy": {"x": rbin, "y": horig},
            },
            **intermediate_results,
        }
        hfit = profile_function(
            crater,
            list(popt_map.keys()),
            rvals,
            None,
            False,
            basicmoon_bindings.basicmoon_profile,
            DummyCrater,
            *popt_map.values(),
        )

        # Check to make sure that we didn't make a crater with a ring as a floor
        hmin = np.min(hfit)
        if crater.floor_elevation < 1.2 * hmin:
            if i == ntries - 1:
                raise RuntimeError(
                    "Could not find a fit. Try adjusting input parameters and try again"
                )
            print(
                f"{i} crater.floor_elevation: {crater.floor_elevation} < {1.2 * hmin}. Trying again."
            )
        else:
            break
    plot_input = {
        "DTM data": {
            "x": rvals,
            "y": hvals,
            "style": "scatter",
            "kwargs": {"marker": ".", "s": 1, "color": "black", "zorder": 1},
        },
        "Binned": {
            "x": rbin,
            "y": hbin,
            "yerr": hstd,
            "style": "errorbar",
            "kwargs": {
                "fmt": ".",
                "elinewidth": 1,
                "markersize": 2,
                "color": colors[1],
                "zorder": 10,
            },
        },
    }
    if save_intermediate:
        for i, (k, v) in enumerate(intermediate_results.items()):
            plot_profile(
                crater=crater,
                filename=plotdir
                / "binned_profiles"
                / f"{crater.label}_profile_intermediate{i:02d}.{plot_extension}",
                plot_input={
                    **plot_input,
                    k: {
                        **v["plot_xy"],
                        "style": "line",
                        "kwargs": {"color": colors[i + 2], "zorder": 100},
                    },
                },
                morphometric_vars=v["morphometric_vars"],
                popt_map=v["popt_map"],
                psig_map=v["psig_map"],
                ymin=ymin,
                ymax=ymax,
            )
    if "floor_elevation" not in popt_map:
        popt_map["floor_elevation"] = crater.floor_elevation
        psig_map["floor_elevation"] = crater.floor_elevation_std

    plot_input["Fit"] = {
        "x": rvals,
        "y": hfit,
        "style": "line",
        "kwargs": {"color": colors[0], "zorder": 30},
    }

    print("Plotting result")
    if plotfile is None:
        filename = (
            plotdir / "binned_profiles" / f"{crater.label}_profile.{plot_extension}"
        )
    else:
        filename = Path(plotfile)
    plot_profile(
        crater=crater,
        filename=filename,
        plot_input=plot_input,
        morphometric_vars=list(popt_map.keys()),
        popt_map=popt_map,
        psig_map=psig_map,
        ymin=ymin,
        ymax=ymax,
    )

    return popt_map, psig_map


def goodness_of_fit(y: float, f: float, p: int):
    """
    Returns Adjusted R**2 of a fit.

    Parameters
    ----------
    y : float
        The data used in the fit.
    f : float
        The values of the fitted function.
    p : int
        The number of degrees of freedom (number of fit variables.)
    """

    n = len(y)
    df_res = n - p - 1
    ss_res = np.sum((y - f) ** 2)

    df_tot = n - 1
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    return 1 - ((ss_res / df_res) / (ss_tot / df_tot))


def ring_fitter(
    crater, fit_vars_orig, rbin, hbin, hstd, fit_bounds, fit_rim, save_intermediate
):

    craterfit = DummyCrater(crater=crater, popt_map={}, psig_map={})
    if crater.nrings > 0:
        popt_candidates = []
        psig_candidates = []
        ir_candidates = []
        r2 = []
        crater_candidates = []
    for cnum in range(crater.nrings + 1):
        fv = [v for v in fit_vars_orig]
        popt_map, psig_map, intermediate_results = fit_profile(
            sim,
            craterfit,
            fv,
            rbin,
            hbin,
            fit_bounds=fit_bounds,
            fit_rings=True,
            fit_rim=fit_rim,
            save_intermediate=save_intermediate,
        )
        if crater.nrings > 0:
            crater_candidates.append(craterfit)
            popt_candidates.append(popt_map)
            psig_candidates.append(psig_map)
            ir_candidates.append(intermediate_results)
            # Check goodness of fit and compare with other solutions with different numbers of rings. Bias this towards key areas, such as the floor and rim
            goodness_range = (rbin <= craterfit.radius / 3.0) | (
                (rbin >= 0.8 * craterfit.radius) & (rbin <= 1.2 * craterfit.radius)
            )
            rim_height_penalty = 1.0
            for i, ring in enumerate(craterfit.rings):
                pre = ring_prefix(i)
                radius = popt_map[f"{pre}_radius"]
                goodness_range = goodness_range | (
                    (rbin >= 0.8 * radius) & (rbin <= 1.2 * radius)
                )
                if popt_map[f"{pre}_rim_height"] < 1e-4 * radius:
                    rim_height_penalty *= 0.9
            rgoodness = rbin[goodness_range]
            hgoodness = hbin[goodness_range]

            hfit = profile_function(
                craterfit,
                list(popt_map.keys()),
                rgoodness,
                None,
                False,
                basicmoon_bindings.basicmoon_profile,
                DummyCrater,
                *popt_map.values(),
            )
            r2.append(
                goodness_of_fit(y=hgoodness, f=hfit, p=len(popt_map))
                * rim_height_penalty
            )

        craterfit = DummyCrater(crater=crater, psig_map={}, popt_map={})
        for j in range(cnum + 1):
            if craterfit.nrings > 0:
                del craterfit.rings[-1]

    if crater.nrings > 0:
        ibest = np.argmax(np.array(r2))
        popt_map = popt_candidates[ibest]
        psig_map = psig_candidates[ibest]
        intermediate_results = ir_candidates[ibest]
        crater = crater_candidates[ibest]
    else:
        crater = DummyCrater(crater=crater, popt_map=popt_map, psig_map=psig_map)
    return crater, popt_map, psig_map, intermediate_results


def get_morphometry_from_radial_slices(
    sim: Simulation,
    crater: BasicMoonCrater | DummyCrater,
    nrbins: int = 500,
    fit_vars: list[str, ...] = default_fit_vars,
    nthetabins: int = 360,
    fit_range: tuple = (None, None),
    plot_extension: str = "png",
    save_intermediate: bool = False,
    fit_rim: bool = True,
    plot_map: bool = True,
):
    """
    Fit crater morphology parameters to a sequence of radial elevation slices.

    Parameters
    ----------
    sim : cratermaker.Simulation
        Active simulation object used to construct crater profiles.
    crater : cratermaker.Crater
        Crater object supplying initial morphology estimates.
    fit_vars : list[str,...]
        List of morphometric variables to fit.
    nrbins : int, optional
        Number of radial bins used to define the radial slices.
    nthetabins : int, optional
        Number of angular bins used to define radial slices.
    fit_range : tuple
        A tuple that gives the range of radius points that will be used for the fit relative to the crater radius. Examples are
        (None, None) : use the full range.
        (None, 1.25) : use all points inward of 1.25 * crater.radius.
        (0.5, 1.5) : use points between 0.5 * crater.radius to 1.5 * crater.radius.
    plot_extension : str, optional
        File extension for the profile plot (e.g., "pdf", "png").
    fit_rim : bool, optional
        If False, but "radius" appears in the fit_vars, then radius will be determined as the radial bin with the highest elevation rather than be fit.
    plot_map : bool, optional
        If true, make draw the rim and floor values on a map of elevation (only for radial style).

    Returns
    -------
    popt_map : dict
        A dict of fit variables as keys and the best-fit parameters from ``scipy.optimize.curve_fit`` as values
    psig_map : dict
        A dict of fit variables as keys and the standard error (sqrt of diagonal of covariance matrix of fitted parameters from ``scipy.optimize.curve_fit``) as values.

    Notes
    -----
    A plot showing the data and the profile fit is written to ``plotdir / f"{crater.label}_profile.{plot_extension}"``.
    """

    crater = DummyCrater(crater=crater, popt_map={}, psig_map={})
    fit_vars_orig = fit_vars.copy()
    rhi = (
        fit_range[1] * crater.radius
        if fit_range[1] is not None
        else sim.surface.local.radius
    )
    region = get_region(sim, crater, fit_range=(None, rhi / crater.radius))
    if "floor_elevation" not in fit_vars:
        print("Computing floor elevation from binned profile...")
        # Use the binned profile to get the floor elevation value and use this for all slices
        _, hbin, hstd = binned_profile(
            region.face_distance, region.desloped_face_elevation, nrbins=nrbins
        )
        ifloor = np.argmin(hbin)
        crater = DummyCrater(
            crater=crater,
            popt_map={"floor_elevation": hbin[ifloor]},
            psig_map={"floor_elevation": hstd[ifloor]},
        )

    uxds = region.uxds.set_coords(["face_distance", "face_bearing"])
    nt = min(int(region.n_face / 100), nthetabins)

    if (region.n_face / nt) > 5 * nrbins:
        thetabin = np.linspace(0, 360, endpoint=True, num=nt + 1)
        rbin = np.linspace(0, rhi, endpoint=False, num=nrbins)

        hgrouped = uxds["desloped_face_elevation"].groupby(
            face_distance=BinGrouper(bins=rbin),
            face_bearing=BinGrouper(bins=thetabin),
        )
        hbin = hgrouped.mean()
        hstd = hgrouped.std()
        hstd = xr.where(hstd < np.abs(1e-3 * hbin), np.abs(1e-3 * hbin), hstd)
        thetabin = np.array([v.mid for v in hbin.face_bearing_bins.data])
    else:
        uxds = uxds.sortby("face_bearing")
        hbin = None
        hstd = None
        thetabin, dtheta = np.linspace(0, 360, endpoint=False, num=nt, retstep=True)
        thetabin += dtheta / 2

    ymin = 1.25 * region.desloped_face_elevation.min()
    ymax = 1.25 * region.desloped_face_elevation.max()

    # Create folders for saving the radial slice profiles and parameters
    radplotdir = plotdir / "radial_profiles" / f"{crater.label}_radial_profiles"
    raddatdir = datadir / "radial_profiles"
    shutil.rmtree(radplotdir, ignore_errors=True)
    radplotdir.mkdir(parents=True)
    raddatdir.mkdir(parents=True, exist_ok=True)

    datfile = raddatdir / f"{crater.label}_radial_profile_fits.csv"

    all_popt_map, all_psig_map = [], []
    isnew = True
    popt_map = {}
    for v in fit_vars:
        popt_map[v] = getattr(crater, v)
    new_rim = np.empty(nt)
    new_rim_std = np.empty_like(new_rim)
    if crater.nrings > 0:
        new_ring = np.empty((nt, crater.nrings))
        new_ring_std = np.empty_like(new_ring)
    new_floor = np.empty_like(new_rim)
    new_floor_std = np.empty_like(new_rim)
    all_radius = []
    crater_orig = DummyCrater(crater=crater, popt_map={}, psig_map={})

    for itheta, theta in enumerate(tqdm(thetabin, desc="Fitting radial slices")):
        if hbin is not None:
            ihvals = hbin[:, itheta]
            goodvals = ~np.isnan(ihvals)
            ihvals = ihvals[goodvals]
            irvals = np.array([v.mid for v in ihvals.face_distance_bins.data])
            ihvals = ihvals.data
            ihstd = hstd[:, itheta][goodvals].data
        else:
            iuxds = uxds.sel(
                face_bearing=slice(theta - dtheta / 2, theta + dtheta / 2)
            ).sortby("face_distance")
            irvals = iuxds.face_distance.data
            ihvals = iuxds.desloped_face_elevation.data[irvals < rhi]
            irvals = irvals[irvals < rhi]
            ihstd = None

        fit_bounds = {}
        if len(irvals) < len(fit_vars):
            continue
        newrad, rim_height = get_rim_from_data(irvals, ihvals, crater)
        radius = newrad if newrad < 1.15 * crater_orig.radius else crater_orig.radius

        crater = DummyCrater(
            crater=crater_orig,
            radius=radius,
            rim_height=rim_height,
            popt_map={},
            psig_map={},
        )
        horig = profile_function(
            crater,
            fit_vars,
            irvals,
            None,
            True,
            basicmoon_bindings.basicmoon_profile,
            DummyCrater,
        )

        all_popt = []
        all_psig = []

        fv = [v for v in fit_vars_orig]

        popt_map, psig_map, intermediate_results = fit_profile(
            sim,
            crater,
            fv,
            irvals,
            ihvals,
            hstd=ihstd,
            fit_bounds=fit_bounds,
            fit_rings=True,
            fit_rim=fit_rim,
            save_intermediate=save_intermediate,
        )
        all_popt.append(popt_map)
        all_psig.append(psig_map)

        for k in list(popt_map.keys()):
            popt_map[k] = np.mean([p[k] for p in all_popt])
            psig_map[k] = np.mean([p[k] for p in all_psig])

        intermediate_results = {
            "Initial guess": {
                "popt_map": crater.as_popt_map,
                "psig_map": crater.as_psig_map,
                "morphometric_vars": fv,
                "plot_xy": {"x": irvals, "y": horig},
            },
            **intermediate_results,
        }
        crater = DummyCrater(crater=crater, popt_map=popt_map, psig_map=psig_map)

        new_rim[itheta] = crater.radius
        new_rim_std[itheta] = crater.radius_std
        if crater.nrings > 0:
            for i, ring in enumerate(crater.rings):
                new_ring[itheta, i] = ring.radius
                if ring.radius_std < (crater.radius - ring.radius) / 2:
                    new_ring_std[itheta, i] = ring.radius_std
                else:
                    new_ring_std[itheta, i] = 0.0
                if i == (crater.nrings - 1):
                    new_floor[itheta] = ring.floor_radius
                    if ring.floor_radius_std < (crater.radius - ring.floor_radius) / 2:
                        new_floor_std[itheta] = ring.floor_radius_std
                    else:
                        new_floor_std[itheta] = 0.0
        else:
            new_floor[itheta] = crater.floor_radius
            if crater.floor_radius_std < (crater.radius - crater.floor_radius) / 2:
                new_floor_std[itheta] = crater.floor_radius_std
            else:
                new_floor_std[itheta] = 0.0

        save_morphometric_parameters(
            filename=datfile,
            crater=crater,
            popt_map=popt_map,
            psig_map=psig_map,
            isnew=isnew,
            additional_data={"theta": theta},
        )
        isnew = False
        thetaint = int(theta)
        filename = (
            radplotdir / f"{crater.label}_profile_{thetaint:03d}.{plot_extension}"
        )
        hfit = profile_function(
            crater,
            list(popt_map.keys()),
            irvals,
            None,
            False,
            basicmoon_bindings.basicmoon_profile,
            DummyCrater,
            *popt_map.values(),
        )
        if hbin is not None:
            datalabel = f"Binned DTM at ϴ = {theta:.1f}°"
        else:
            datalabel = f"DTM at ϴ = {theta:.1f}°"
        plot_input = {
            datalabel: {
                "x": irvals,
                "y": ihvals,
                "style": "scatter",
                "kwargs": {"marker": "o", "s": 10, "color": "black", "zorder": 10},
            },
        }
        if save_intermediate:
            for i, (k, v) in enumerate(intermediate_results.items()):
                plot_profile(
                    crater=crater,
                    filename=radplotdir
                    / f"{crater.label}_profile_{thetaint:03d}_intermediate{i:02d}.{plot_extension}",
                    plot_input={
                        **plot_input,
                        k: {
                            **v["plot_xy"],
                            "style": "line",
                            "kwargs": {"color": colors[i + 2], "zorder": 100},
                        },
                    },
                    morphometric_vars=v["morphometric_vars"],
                    popt_map=v["popt_map"],
                    psig_map=v["psig_map"],
                    ymin=ymin,
                    ymax=ymax,
                )
        plot_input["Fit"] = {
            "x": irvals,
            "y": hfit,
            "style": "line",
            "kwargs": {"color": colors[0], "zorder": 30},
        }
        morphometric_vars = list(popt_map.keys())
        if "floor_elevation" not in morphometric_vars:
            morphometric_vars.append("floor_elevation")
        if "radius" not in morphometric_vars:
            morphometric_vars.append("radius")

        plot_profile(
            crater=crater,
            filename=filename,
            plot_input=plot_input,
            morphometric_vars=morphometric_vars,
            popt_map=popt_map,
            psig_map=psig_map,
            ymin=ymin,
            ymax=ymax,
        )
        all_popt_map.append(popt_map)
        all_psig_map.append(psig_map)
        all_radius.append(crater.radius)

    # Compute the mean parameter values and covariance across all the radial slices to get a more robust estimate of the crater morphology parameters and their uncertainties. This accounts for both the within-slice covariance (from curve_fit) and the between-slice covariance (from variability in best-fit parameters across slices).
    all_popt = {}
    all_psig = {}
    for popt_map, psig_map in zip(all_popt_map, all_psig_map, strict=True):
        for k, v in popt_map.items():
            if k not in all_popt:
                all_popt[k] = [v]
                all_psig[k] = [psig_map[k]]
            else:
                all_popt[k].append(v)
                all_psig[k].append(psig_map[k])

    popt_mean_map = {}
    psig_mean_map = {}
    for k, v in all_popt.items():
        popt_mean_map[k] = np.mean(v)
        psig_mean_map[k] = np.std(v)

    crater = DummyCrater(crater=crater, psig_map=psig_mean_map, popt_map=popt_mean_map)
    geod = Geod(a=sim.target.radius, b=sim.target.radius)
    lon, lat = crater.location

    def polymaker(r, r_std):
        polys = []
        loval = 1e-3 * crater.radius
        for i in range(4):
            rlo = r[:] - i * r_std[:]
            rhi = r[:] + i * r_std[:]
            if i == 0:
                rlo -= loval
                rhi += loval
            negval = rlo < loval
            if any(negval):
                rlo[negval] = loval

            lon_lo, lat_lo, _ = geod.fwd(
                lon * np.ones_like(thetabin),
                lat * np.ones_like(thetabin),
                thetabin,
                rlo,
            )
            lon_hi, lat_hi, _ = geod.fwd(
                lon * np.ones_like(thetabin),
                lat * np.ones_like(thetabin),
                thetabin,
                rhi,
            )

            outer_poly = Polygon(zip(lon_hi, lat_hi, strict=True))
            if not outer_poly.is_valid:
                outer_poly = make_valid(outer_poly)
            inner_poly = Polygon(zip(lon_lo, lat_lo, strict=True))
            if not inner_poly.is_valid:
                inner_poly = make_valid(inner_poly)
            polys.append(difference(outer_poly, inner_poly))
        return polys

    rim_polys = polymaker(new_rim[:], new_rim_std[:])
    floor_polys = polymaker(new_floor[:], new_floor_std[:])
    ring_polys = []
    if crater.nrings > 0:
        for i in range(crater.nrings):
            ring_polys.extend(polymaker(new_ring[:, i], new_ring_std[:, i]))

    gs_rim = gpd.GeoSeries(rim_polys, crs=sim.surface.crs).to_crs(sim.surface.crs)
    rimfile = vectorfiledir / "rim" / f"{crater.label}_v2.gpkg"
    if rimfile.exists():
        rimfile.unlink()
    gs_rim.to_file(
        filename=rimfile,
        driver="GPKG",
        layer="rim",
    )

    gs_floor = gpd.GeoSeries(floor_polys, crs=sim.surface.crs).to_crs(sim.surface.crs)
    floorfile = vectorfiledir / "floor" / f"{crater.label}_v2.gpkg"
    if floorfile.exists():
        floorfile.unlink()
    gs_floor.to_file(floorfile, driver="GPKG", layer="floor")

    if crater.nrings > 0:
        gs_ring = gpd.GeoSeries(ring_polys, crs=sim.surface.crs).to_crs(sim.surface.crs)
        ringfile = vectorfiledir / "ring" / f"{crater.label}_v2.gpkg"
        if ringfile.exists():
            ringfile.unlink()
        gs_ring.to_file(ringfile, driver="GPKG", layer="floor")

    # Save the centroid of the rim as the updated location
    location = tuple(
        gs_rim.to_crs(sim.surface.local.crs)
        .centroid.to_crs(sim.surface.crs)
        .get_coordinates()
        .to_numpy()[0]
    )
    popt_mean_map["location"] = location
    psig_mean_map["location"] = (0, 0)

    if plot_map:
        print("Plotting elevation data...")
        import cmasher as cmr

        cmap = cmr.eclipse
        cmcol = cmap.colors
        ncol = len(cmcol)
        for i, c in enumerate(cmcol):
            cmcol[i] = [float(v) for v in list(np.array(c) + 0.1 * (ncol - i) / ncol)]
        ax = sim.surface.local.plot(
            variable_name="desloped_face_elevation",
            plot_style="hillshade",
            show=False,
            save=False,
            close_when_done=False,
            solar_angle=30,
            cmap=cmap,
            label=f"D = {format_large_units(crater.diameter, quantity='length')} ({crater.morphology_type.title()})",
        )

        gs_rim.to_crs(sim.surface.local.crs).plot(
            ax=ax,
            color=colors[0],
            alpha=0.25,
        )

        gs_floor.to_crs(sim.surface.local.crs).plot(
            ax=ax,
            color=colors[1],
            alpha=0.25,
        )
        if crater.nrings > 0:
            gs_ring.to_crs(sim.surface.local.crs).plot(
                ax=ax,
                color=colors[2],
                alpha=0.25,
            )
        fig = ax.get_figure()
        fig.savefig(plotdir / "maps" / "topography" / f"{crater.label}_map.png")
        plt.close(fig)

    if "radius" not in popt_mean_map:
        popt_mean_map["radius"] = np.mean(all_radius)
        psig_mean_map["radius"] = np.std(all_radius)
    if "floor_elevation" not in popt_mean_map:
        popt_mean_map["floor_elevation"] = crater.floor_elevation
        psig_mean_map["floor_elevation"] = crater.floor_elevation_std

    fit_vars = fit_vars_orig.copy()

    return popt_mean_map, psig_mean_map


def get_sim(
    crater: BasicMoonCrater,
    name_append: str | None = None,
    morphology: str = "basicmoon",
    surface: str = "datasurface",
    pix: int | None = None,
    region_size_factor: float = _REGION_SIZE_FACTOR,
    reset_surface: bool = False,
):
    """
    Get a cratermaker Simulation object for a given crater, using the DEM files if provided.

    If the simulation directory exists, then the surface will be read in from the saved files. If not, then a new surface will be generated using a local radius given by _REGION_SIZE_FACTOR * crater.radius.

    Parameters
    ----------
    crater : BasicMoonCrater
        Crater object for which to create a simulation.
    name_append : str, optional
        String to append to the simulation directory name.
    morphology : str, optional
        The Cratermaker Morphology model to use. Default is "basicmoon".
    surface : str, optional
        The Cratermaker Surface model to use. Default is "datasurface".
    pix : int, optional
        The pix value of the Cratermaker surface. If None, then the value will be determined based on whether or not an explicit dem_file_list is passed or not. If it is, then that file's resolution is used. If not, then the pix value is _REGION_SIZE_FACTOR * crater.radius / 1000.
    region_size_factor: float, optional
        A scaling factor that scales the local region size of the surface as a multiple of the input crater radius. Default is the constant _REGION_SIZE_FACTOR.
    reset_surface: bool, optional
        If True, it will reset the surface based on the parameters passed to the function. If False, it will check if the simulation directory exists or not, and reset the surface only if it doesn't. When crater properties change between runs, resetting the surface can be very expensive, so this can reduce the amount of time it takes to do iterative fits to crater morphometries.

    Returns
    -------
    cratermaker.Simulation
        Simulation object for the given crater.
    """

    if name_append is not None:
        simdir = simrootdir / f"{crater.label}_{name_append}"
    else:
        simdir = simrootdir / f"{crater.label}"

    if reset_surface or not simdir.exists():
        local_location = crater.location
        local_radius = region_size_factor * crater.radius
        # Compute approximate pixel value to get us a good surface for measuring
        dem_file_list = (
            crater.dem_file_list.copy()
            if hasattr(crater, "dem_file_list") and type(crater.dem_file_list) is list
            else None
        )
        if pix is None:
            if surface == "hireslocal" or dem_file_list is None:
                pix = max(local_radius / 500, 2.0)
            else:
                pix = None
        # Do a first pass to get resolution information
        sleep(10)  # Short delay to prevent 429 errors at PDS
        sim = Simulation(
            simdir=simdir,
            surface=surface,
            morphology=morphology,
            local_location=local_location,
            local_radius=local_radius,
            pix=pix,
            ask_overwrite=False,
            reset=True,
            dem_file_list=dem_file_list,
            rng_seed=109810232,
        )
        if sim.surface.component_name == "hireslocal":
            lon_min, lon_max, lat_min, lat_max = sim.surface.get_location_extents(
                sim.surface.local_location, sim.surface.local_radius
            )
            lores_dem_file_list, _ = sim.surface.data_composer().get_lola_dem_file_list(
                pix=pix, lat_range=(lat_min, lat_max), lon_range=(lon_min, lon_max)
            )
            sim.surface.local.update_elevation(lores_dem_file_list)

            if dem_file_list is not None and any(
                lo != hi
                for lo, hi in zip(dem_file_list, lores_dem_file_list, strict=False)
            ):
                sim.surface.local.update_elevation(
                    dem_file_list,
                )
            sim.save()
        if (
            sim.surface.component_name == "datasurface"
            and crater.diameter / sim.surface.pix < 50
        ):
            print(
                "This is a low resolution surface, so we will expand the local_radius to capture more pixels"
            )
            local_radius *= 2
            sim = Simulation(
                simdir=simdir,
                surface=surface,
                morphology=morphology,
                local_location=local_location,
                local_radius=local_radius,
                pix=pix,
                ask_overwrite=False,
                reset=True,
                dem_file_list=dem_file_list,
                rng_seed=109810232,
            )
        sim.save()
    else:
        sim = Simulation(
            simdir=simdir,
            ask_overwrite=False,
            reset=False,
        )

    sim.counting.reset()
    if isinstance(crater, DummyCrater):
        crater = crater.to_cratermaker(sim)
    sim.counting.add(crater=crater)
    return sim


def get_region(sim, crater, fit_range):
    maxr = (
        fit_range[1] * crater.radius
        if fit_range[1] is not None
        else _REGION_SIZE_FACTOR * crater.radius
    ) * 1.1
    print(
        f"Extracting {crater.name} region of radius {maxr / crater.radius:.2f} R_c and computing reference plane..."
    )
    region = sim.surface.extract_region(location=crater.location, region_radius=maxr)
    region.compute_desloped_face_elevation()
    region.add_data(
        name="face_bearing",
        data=region.face_bearing,
        long_name="bearing angle of face",
        units="degrees",
        isfacedata=True,
        overwrite=True,
    )
    region.add_data(
        name="face_distance",
        data=region.face_distance,
        long_name="distance of face from local location",
        units="m",
        isfacedata=True,
        overwrite=True,
    )

    return region


def get_ring_popt_maps(
    fit_vars: list[str], popt: list[float], psig: list[float] | None = None
):
    """
    Process the popt and psig lists to create dictionaries, and also extract any ring values as list of dicts.

    Parameters
    ----------
    fit_vars : list[str]
        A list of variables to use as keys. Ring variables will be prepended by "ring0N_" where N is the ring number from 1.
    popt : list[float]
        The values of the parameters to include in the returned popt_map.
    psig : list[float], optional
        The values of the standard errors to include in the returned psig_map.
    Returns
    -------
    popt_map : dict
        A mapping of primary crater parameters.
    psig_map : dict | None
        A mapping of primary crater standard errors. None if psig was not input.
    ring_popt_maps : list[dict] | None
        A list of ring parameter maps. None if the crater object has no rings
    ring_psig_maps : list[dict] | None
        A list of ring parameter maps. None if the crater object has no rings or psig was not input.
    """
    popt_map = dict(zip(fit_vars, popt))
    if psig is not None:
        psig_map = dict(zip(fit_vars, psig))
    else:
        psig_map = None

    if any("ring" in v and "peak" not in v for v in fit_vars):
        ring_nums = []
        num_rings = 0
    else:
        return popt_map, psig_map, None, None

    for v in fit_vars:
        if "ring" in v and "peak" not in v:
            n = int(v.replace("ring", "").split("_")[0])
            if n not in ring_nums:
                ring_nums.append(n)
                num_rings += 1

    ring_popt_maps = [{}]
    if psig is not None:
        ring_psig_maps = [{}]
    else:
        ring_psig_maps = None

    for v in fit_vars:
        if "ring" in v and "peak" not in v:
            n = int(v.replace("ring", "").split("_")[0])
            while n > len(ring_popt_maps):
                ring_popt_maps.append({})
                if psig is not None:
                    ring_psig_maps.append({})
            var = v.replace(f"ring{n:02d}_", "")
            ring_popt_maps[n - 1][var] = popt_map.pop(v)
            if psig is not None and v in psig_map:
                ring_psig_maps[n - 1][var] = psig_map.pop(v)

    return popt_map, psig_map, ring_popt_maps, ring_psig_maps


def profile_function(
    crater: DummyCrater,
    fit_vars: list[str, ...],
    radial_distances: np.ndarray,
    bearings: np.ndarray | None,
    orig: bool,
    pfunc: callable,
    DummyCrater: type[DummyCrater],
    *popt: list[float, ...],
):
    """
    Evaluate the crater radial elevation profile for a parameter vector.

    Parameters
    ----------
    crater : DummyCrater
        Crater object supplying initial morphology estimates. The parameter vector in ``popt`` is mapped to crater morphology parameters using the ``popt_to_variable_map``, and a DummyCrater object is constructed from these parameters and the input crater to supply the necessary attributes for constructing the crater profile.
    fit_vars : list[str,...]
        List of morphometric variables corresponding to the parameter vector in ``popt``. This is used to map the optimized parameters in ``popt`` to crater morphology parameters for constructing the crater profile.
    radial_distances : numpy.ndarray
        Radial distances (m) at which to evaluate the profile.
    bearings : numpy.ndarray
        Bearing angles (deg) at which to evaluate the profile.
    orig: bool
    pfunc : callable,
        The profile function for the crater
    DummyCrater : type[DummyCrater]
        The DummyCrater class to use to cast the fit parameters to the crater generation
    popt : list[float,...]
        List of optimized crater morphology parameters.

    Returns
    -------
    numpy.ndarray
        Modeled crater elevation profile evaluated at ``radial_distances``.
    """
    if bearings is None:
        bearings = np.zeros_like(radial_distances)
    else:
        bearings = np.radians(bearings)
    reference_elevations = np.zeros_like(radial_distances)

    if not orig:
        popt_map = dict(zip(fit_vars, popt))
        crater = DummyCrater(crater=crater, popt_map=popt_map, psig_map={})

    profile = pfunc(
        radial_distances=radial_distances,
        bearings=bearings,
        reference_elevations=reference_elevations,
        crater=crater,
        rings=crater.rings,
        include_crater=True,
        include_ejecta=True,
    )

    return profile


def get_rim_from_data(r, h, crater):
    rangesize = 0.25
    while rangesize <= 1.0:
        rim_range = (r >= (1.0 - rangesize) * crater.radius) & (
            r <= (1.0 + rangesize) * crater.radius
        )
        if np.all(~rim_range):
            rangesize += 0.25
        else:
            break
    imax = np.argmax(h[rim_range])
    radius = r[rim_range][imax]
    rim_height = h[rim_range][imax]
    return radius, rim_height


def fit_profile(
    sim: Simulation,
    crater: BasicMoonCrater,
    fit_vars: list[str, ...],
    r: np.ndarray,
    h: np.ndarray,
    hstd: np.ndarray = None,
    p0_map: dict[str, float] | None = None,
    fit_bounds: dict[str, tuple[float, float]] = {},
    fit_rings: bool = True,
    fit_rim: bool = True,
    save_intermediate: bool = False,
):
    """
    Fit crater morphology parameters to a radial elevation profile.

    Parameters
    ----------
    sim : cratermaker.Simulation
        Active simulation object used to construct crater profiles.
    crater : cratermaker.Crater
        Crater object supplying initial morphology estimates.
    fit_vars : list[str,...]
        List of morphometric variables to fit.
    r : numpy.ndarray
        Radial distances (m) for the observed profile.
    h : numpy.ndarray
        Observed elevations (m) corresponding to ``r``.
    hstd : numpy.ndarray, optional
        Observed standard deviations of the elevation data (m) corresponding to ``r``.
    p0_map : dict[str, float], optional
        Initial guess for the parameter vector. If ``None``, initial guesses are derived from crater size and morphology.
    fit_bounds : dict[str, tuple[float,float]], optional
        A dict used to override the default fit bounds on individual parameters. Keys are parameters, and value is tuple of lower and upper bounds.
    fit_rings : bool = True
        Indicate whether or not rings should be fit.
    fit_rim : bool = True
        Indicates whether to fit radius and rim_height.
    save_intermediate : bool = False

    Returns
    -------
    popt_map : dict
        A dict of fit variables as keys and the best-fit parameters from ``scipy.optimize.curve_fit`` as values
    psig_map : dict
        A dict of fit variables as keys and the covariance matrix of fitted parameters from ``scipy.optimize.curve_fit`` as values.
    intermediate_results : dict
        A dict of intermediate fit results for each step if requested (otherwise empty)

    Notes
    -----
    Parameter bounds are derived from crater size and initial morphology values.
    Initial guesses outside bounds are clipped before fitting.
    """
    fit_vars_orig = fit_vars.copy()

    intermediate_results = {}

    def _set_default_fit_bounds(crater, fit_vars, pre=None):
        default_fit_bounds = {}
        if crater.nrings > 0 and fit_rings:
            rhi = crater.radius
            for i, ring in enumerate(crater.rings):
                pre = ring_prefix(i)
                radius_init = ring.radius
                if p0_map is not None:
                    radius_init = p0_map.get(f"{pre}_radius", radius_init)

                ring_bounds = _set_default_fit_bounds(ring, fit_vars, pre)

                if i == crater.nrings - 1:
                    rlo = 0.0
                    lo = crater.radius / 3.0
                else:
                    rlo = crater.rings[i + 1].radius

                hi = ring.radius + (rhi - ring.radius) / 3.0
                lo = ring.radius - (ring.radius - rlo) / 3.0

                ring_bounds[f"{pre}_radius"] = (lo, hi)
                default_fit_bounds = {**default_fit_bounds, **ring_bounds}
                rhi = ring.radius

            pre = None

        radmin = 0.75 * crater.radius
        radmax = 1.25 * crater.radius
        default_fit_bounds = {
            "radius": (radmin, radmax),
            "floor_elevation": (
                1.2 * crater.floor_elevation,
                0.8 * crater.floor_elevation,
            ),
            "floor_radius": (0.0, 0.99 * radmax),
            "wall_curvature": (0.0, 1.0),
            "floor_blend": (0.0, 1.0),
            "rim_width": (1e-2 * radmin, radmax),
            "rim_height": (0.0 * radmax, radmax),
            "ejprofile": (-6.0, -0.25),
            "frac_ejrim": (0.1, 0.9),
            "peak_height": (0.0, 0.2 * radmax),
            "peak_width": (0.0, 0.4 * radmax),
            "peak_ring_radius": (
                0.0,
                2 * (crater.floor_radius + crater.peak_width),
            ),
            **default_fit_bounds,
        }

        # If the crater contains std information, use this to set bounds to be +/- 3*std (unless that exceeds the default bounds)
        for k in valid_fit_vars:
            k_std = f"{k}_std"
            if hasattr(crater, k_std):
                vstd = getattr(crater, k_std)
                if vstd > 1e-8:
                    v = getattr(crater, k)
                    lo = max(v - 3 * vstd, default_fit_bounds[k][0])
                    hi = min(v + 3 * vstd, default_fit_bounds[k][1])
                    default_fit_bounds[k] = (lo, hi)

        for k in list(default_fit_bounds.keys()):
            lo = default_fit_bounds[k][0]
            hi = default_fit_bounds[k][1]
            if lo > hi:
                default_fit_bounds[k] = (hi, lo)
            elif lo == hi:
                default_fit_bounds[k] = (lo - 1e-8, hi + 1e-8)
        if pre is None:
            return default_fit_bounds
        else:
            return {f"{pre}_{k}": v for k, v in default_fit_bounds.items()}

    def _one_fit(craterfit, p0_map, fit_bounds, varlist, r, h, hstd):

        lower_bounds = {k: fit_bounds[k][0] for k in varlist if k in fit_bounds}
        upper_bounds = {k: fit_bounds[k][1] for k in varlist if k in fit_bounds}

        for k, v in p0_map.items():
            if k not in lower_bounds:
                continue
            if v < lower_bounds[k] or v > upper_bounds[k]:
                # print(
                #     f"Warning: Initial guess for parameter {k} is out of bounds. Adjusting to be within bounds: {lower_bounds[k]} < {p0_map[k]} < {upper_bounds[k]}"
                # )
                p0_map[k] = np.clip(
                    p0_map[k],
                    min(lower_bounds[k] + 1e-12, upper_bounds[k]),
                    max(upper_bounds[k] - 1e-12, lower_bounds[k]),
                )

        p0 = [p0_map[k] for k in varlist]
        bounds = (
            [lower_bounds[k] for k in lower_bounds],
            [upper_bounds[k] for k in upper_bounds],
        )
        absolute_sigma = hstd is not None
        popt, pcov, infodict, mesg, ier = curve_fit(
            lambda xdata, *params: profile_function(
                craterfit,
                varlist,
                xdata,
                None,
                False,
                basicmoon_bindings.basicmoon_profile,
                DummyCrater,
                *params,
            ),
            r,
            h,
            sigma=hstd,
            p0=p0,
            absolute_sigma=absolute_sigma,
            maxfev=10000000,
            xtol=1e-6,
            ftol=1e-6,
            bounds=bounds,
            full_output=True,
        )
        popt_map = dict(zip(varlist, popt))
        pcov_map = dict(zip(varlist, pcov))
        psig_map = dict(zip(pcov_map.keys(), np.sqrt(np.diag(list(pcov_map.values())))))
        return popt_map, psig_map

    if p0_map is None or len(p0_map) == 0:
        fit_init = {}
        for k in fit_vars:
            fit_init[k] = getattr(crater, k)

        if crater.nrings > 0 and fit_rings:
            ring_init = {}
            for i, ring in enumerate(crater.rings):
                pre = ring_prefix(i)
                # Add variables that can float for rings even if they are fixed for the main crater object
                for v in ["radius", "rim_height", "rim_width"]:
                    rv = f"{pre}_{v}"
                    if v not in fit_init:
                        ring_init[rv] = getattr(ring, v)
                        fit_vars.append(rv)
                for k, v in fit_init.items():
                    # Remove variables that are linked to the main crater object
                    if k in [
                        "floor_elevation",
                        "peak_height",
                        "peak_width",
                        "peak_ring_radius",
                        "floor_blend",
                        "frac_ejrim",
                        "ejprofile",
                    ]:
                        continue
                    ring_init[f"{pre}_{k}"] = getattr(ring, k)
                    if k in fit_vars:
                        fit_vars.append(f"{pre}_{k}")
            fit_init = {**fit_init, **ring_init}

        p0_map = {k: fit_init[k] for k in fit_vars}

    # Remove fit variables defined with 0 std
    for k, v in crater.as_psig_map.items():
        if v < 1e-8 and k in fit_vars:
            fit_vars.remove(k)

    default_fit_bounds = _set_default_fit_bounds(crater, fit_vars)

    fit_bounds = {**default_fit_bounds, **fit_bounds}

    # First fit the region around the rim
    valid_rim_vars = ["rim_width", "radius", "frac_ejrim", "ejprofile", "floor_radius"]
    if fit_rim:
        valid_rim_vars.extend(["radius", "rim_height"])

    rim_fit_vars = [k for k in fit_vars if k in valid_rim_vars]
    if crater.nrings > 0 and fit_rings:
        for j, jring in enumerate(crater.rings):
            if i != j:
                jpre = ring_prefix(j)
                rim_fit_vars.append(f"{jpre}_floor_radius")

    psig_map = {}
    craterfit = DummyCrater(crater=crater, popt_map=p0_map, psig_map={})
    if len(rim_fit_vars) > 0:
        if hasattr(crater, "radius_std"):
            rlo = crater.radius - 3 * crater.radius_std
            rhi = crater.radius + 2 * crater.radius_std
        else:
            rlo = 0.9 * crater.radius
            rhi = 1.1 * crater.radius
        inrange = (r > rlo) & (r < rhi)
        rrim = r[inrange]
        if len(rrim) > len(rim_fit_vars):
            hrim = h[inrange]
            hrimstd = None if hstd is None else hstd[inrange]

            rim_popt_map, rim_psig_map = _one_fit(
                craterfit, p0_map, fit_bounds, rim_fit_vars, rrim, hrim, hrimstd
            )

            p0_map = {**p0_map, **rim_popt_map}
            psig_map = {**psig_map, **rim_psig_map}
            if save_intermediate:
                intermediate_results["rim"] = {
                    "popt_map": rim_popt_map,
                    "psig_map": rim_psig_map,
                    "morphometric_vars": rim_fit_vars,
                    "plot_xy": {
                        "x": rrim,
                        "y": profile_function(
                            craterfit,
                            rim_fit_vars,
                            rrim,
                            None,
                            False,
                            basicmoon_bindings.basicmoon_profile,
                            DummyCrater,
                            *rim_popt_map.values(),
                        ),
                    },
                }

    # Next, fit the region around the floor
    valid_floor_vars = [
        "floor_blend",
        "peak_height",
        "peak_width",
        "peak_ring_radius",
    ]
    if craterfit.nrings > 0:
        valid_floor_vars.append(f"{ring_prefix(craterfit.nrings)}_floor_radius")
    else:
        valid_floor_vars.append("floor_radius")
    floor_fit_vars = [k for k in fit_vars if k in valid_floor_vars]
    if len(floor_fit_vars) > 0:
        craterfit = DummyCrater(crater=craterfit, popt_map=p0_map, psig_map={})
        if craterfit.nrings > 0:
            floor_radius = craterfit.rings[-1].floor_radius
        else:
            floor_radius = craterfit.floor_radius

        cutoff = max(
            1.5 * floor_radius + craterfit.peak_ring_radius + 2 * craterfit.peak_width,
            0.2 * craterfit.radius,
        )

        inrange = r < cutoff
        rfloor = r[inrange]
        if len(rfloor) > len(floor_fit_vars):
            hfloor = h[inrange]
            hfloorstd = None if hstd is None else hstd[inrange]

            floor_popt_map, floor_psig_map = _one_fit(
                craterfit, p0_map, fit_bounds, floor_fit_vars, rfloor, hfloor, hfloorstd
            )

            p0_map = {**p0_map, **floor_popt_map}
            psig_map = {**psig_map, **floor_psig_map}
            if save_intermediate:
                intermediate_results["floor"] = {
                    "popt_map": floor_popt_map,
                    "psig_map": floor_psig_map,
                    "morphometric_vars": floor_fit_vars,
                    "plot_xy": {
                        "x": rfloor,
                        "y": profile_function(
                            craterfit,
                            floor_fit_vars,
                            rfloor,
                            None,
                            False,
                            basicmoon_bindings.basicmoon_profile,
                            DummyCrater,
                            *floor_popt_map.values(),
                        ),
                    },
                }

    # Next, fit rings, if any exist
    if craterfit.nrings > 0 and fit_rings:
        for i, ring in enumerate(craterfit.rings):
            pre = ring_prefix(i)
            ring_fit_vars = [v for v in fit_vars if pre in v]
            # floor radii of the main crater and other rings can have an effect on this ring's properties

            ring_fit_vars.append("floor_radius")
            for j, jring in enumerate(craterfit.rings):
                if i != j:
                    jpre = ring_prefix(j)
                    ring_fit_vars.append(f"{jpre}_floor_radius")

            lo, hi = fit_bounds[f"{pre}_radius"]
            inrange = (r > lo) & (r < hi)
            rring = r[inrange]
            if len(rring) > len(ring_fit_vars):
                hring = h[inrange]
                hringstd = None if hstd is None else hstd[inrange]

                craterfit = DummyCrater(crater=craterfit, popt_map=p0_map, psig_map={})
                ring_popt_map, ring_psig_map = _one_fit(
                    craterfit, p0_map, fit_bounds, ring_fit_vars, rring, hring, hringstd
                )

                p0_map = {**p0_map, **ring_popt_map}
                psig_map = {**psig_map, **ring_psig_map}
                if save_intermediate:
                    intermediate_results[pre] = {
                        "popt_map": ring_popt_map,
                        "psig_map": ring_psig_map,
                        "morphometric_vars": ring_fit_vars,
                        "plot_xy": {
                            "x": rring,
                            "y": profile_function(
                                craterfit,
                                ring_fit_vars,
                                rring,
                                None,
                                False,
                                basicmoon_bindings.basicmoon_profile,
                                DummyCrater,
                                *ring_popt_map.values(),
                            ),
                        },
                    }

    # Now fit everything else, using the std values from the first two fits as bounds
    craterfit = DummyCrater(crater=craterfit, popt_map=p0_map, psig_map=psig_map)

    full_fit_vars = list(p0_map.keys())
    if not fit_rim:
        full_fit_vars.remove("radius")
        full_fit_vars.remove("rim_height")

    fit_bounds = _set_default_fit_bounds(craterfit, full_fit_vars)
    popt_map, psig_map = _one_fit(
        craterfit, p0_map, fit_bounds, full_fit_vars, r, h, hstd
    )

    craterfit = DummyCrater(crater=crater, popt_map=popt_map, psig_map=psig_map)
    final_popt = craterfit.as_popt_map

    # Ensure that the final crater values are consistent (some parameters get automatically adjusted upon crater creation)
    popt_map = {k: final_popt[k] for k in popt_map.keys()}

    fit_vars = fit_vars_orig.copy()
    required_vars = ["radius", "rim_height", "rim_width"]
    for v in required_vars:
        if v not in popt_map:
            popt_map[v] = getattr(craterfit, v)
            psig_map[v] = 0.0
    return popt_map, psig_map, intermediate_results


def plot_profile(
    crater: BasicMoonCrater,
    plot_input: dict,
    filename: Path | None = None,
    morphometric_vars: list[str] | None = None,
    popt_map: dict[str, float] | None = None,
    psig_map: dict[str, float] | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
):
    """
    Generate a plot of the crater profile data and fit.

    Parameters
    ----------
    filename : Path
        Output file path for the profile plot.
    plot_input : dict
        Dictionary specifying the data and fit to plot. Each key is a label for the legend, and each value is a dictionary with keys "x", "y", "style" (either "scatter" or "line"), and "kwargs" (a dictionary of keyword arguments to pass to the plotting function).
    morphometric_vars : list[str], optional
        List of morphometric variable names to annotate on the plot. If ``None``, no annotation is added.
    popt_map : dict[str, float], optional
        Parameter map corresponding to the morphometric variables. This is used to extract the values of the specified morphometric variables for annotation. If ``None``, no annotation is added.
    psig_map : dict[str, float], optional
        Parameter map corresponding to the morphometric variable standard error. This is used to extract the values of the specified morphometric variables for annotation. If ``None``, no annotation is added.
    """
    _, ax = plt.subplots(figsize=(8, 3), layout="constrained")
    crater = DummyCrater(crater=crater, popt_map=popt_map, psig_map=psig_map)

    xmax = 2.0 * crater.radius
    ax.set_xlim(0.0, xmax)
    if ymin is None:
        ymin = -0.5 * crater.radius
    if ymax is None:
        ymax = 0.20 * crater.radius
    ax.set_ylim(ymin, ymax)
    if crater.radius < 100e3:
        ax.set_aspect("equal")
    else:
        ax.set_aspect("auto")

    ax.set_xlabel("Radial distance (m)")
    ax.set_ylabel("Elevation (m)")

    ax.set_title(f"{crater.name} - {crater.morphology_type}")
    for label, plot_options in plot_input.items():
        x, y = plot_options["x"], plot_options["y"]
        if plot_options["style"] == "scatter":
            ax.scatter(x, y, label=label, **plot_options["kwargs"])
        elif plot_options["style"] == "line":
            ax.plot(x, y, label=label, **plot_options["kwargs"])
        elif plot_options["style"] == "errorbar":
            yerr = plot_options["yerr"]
            ax.errorbar(x, y, yerr=yerr, label=label, **plot_options["kwargs"])
        if max(y) > ymax:
            ymax *= 1.1
            ax.set_ylim(ymin, ymax)
        if min(y) < ymin:
            ymin *= 1.1
            ax.set_ylim(ymin, ymax)

    if "Fit" in plot_input:
        rvals = plot_input["Fit"]["x"]
        hvals = plot_input["Fit"]["y"]

        def std_box(var, rmask, color, style, yoffset=0.0):
            v = crater.as_popt_map[var]

            if style == "vertical":
                ax.vlines(
                    x=v,
                    ymin=ymin,
                    ymax=hvals[np.argmin(np.abs(rvals - v))],
                    linestyle=":",
                    color=color,
                    linewidth=1,
                )
            elif style == "horizontal":
                if len(rvals[rmask]) == 0:
                    xmin = rvals[0]
                    xmax = rvals[-1]
                else:
                    xmin = rvals[rmask][0]
                    xmax = rvals[rmask][-1]
                ax.hlines(
                    y=v + yoffset,
                    xmin=xmin,
                    xmax=xmax,
                    linestyle=":",
                    color=color,
                    linewidth=1,
                )

            if var in crater.as_psig_map:
                v_std = crater.as_psig_map[var]

                for i in range(1, 4):
                    if style == "vertical":
                        rlo = max(v - i * v_std, -1e-8)
                        rhi = v + i * v_std
                        x = np.linspace(rlo, rhi, 20)
                        inrange = (rvals >= rlo) & (rvals <= rhi)
                        if len(rvals[inrange]) == 0:
                            i = np.argmin(np.abs(rvals - v))
                            if i == 0:
                                ilo = i
                            else:
                                ilo = i - 1
                            if i == len(rvals) - 1:
                                ihi = len(rvals)
                            else:
                                ihi = i + 1
                            rinrange = rvals[ilo:ihi]
                            yinrange = hvals[ilo:ihi]
                        else:
                            rinrange = rvals[inrange]
                            yinrange = hvals[inrange]
                        y1 = np.empty_like(x)
                        y2 = np.full_like(y1, ymin)
                        for i in range(y1.size):
                            # Use endpoints if the data don't extend beyond the box limits.
                            # Use nearest neighbor otherwise
                            if rinrange[0] > x[i]:
                                y1[i] = yinrange[0]
                            elif rinrange[-1] < x[i]:
                                y1[i] = yinrange[-1]
                            else:
                                y1[i] = yinrange[np.argmin(np.abs(rinrange - x[i]))]
                    elif style == "horizontal":
                        x = rvals[rmask]
                        y1 = np.full_like(rvals[rmask], v + v_std) + yoffset
                        y2 = np.full_like(rvals[rmask], v - v_std) + yoffset
                    ax.fill_between(
                        x, y1, y2, color=color, alpha=0.25, zorder=1, linewidth=0
                    )

        rmax = max(rvals)
        xmax = 1.1 * rmax
        ax.set_xlim(0.0, xmax)

        # Pre-impact surface line
        inner = rvals < crater.radius
        r_inner_wall = rvals[inner][np.argmin(np.abs(hvals[inner]))]
        ax.hlines(
            y=0.0,
            xmin=r_inner_wall,
            xmax=xmax,
            linestyle="--",
            color="black",
            linewidth=0.2,
        )
        # ax.text(rmax, 0, " Surface", va="bottom", ha="left", fontsize=7)

        std_box(var="radius", rmask=None, color=colors[3], style="vertical")
        std_box(
            var="rim_height",
            rmask=rvals >= crater.radius,
            color=colors[3],
            style="horizontal",
        )
        ax.text(
            rmax,
            crater.rim_height,
            " Rim",
            va="center",
            ha="left",
            fontsize=7,
            color=colors[3],
        )

        floor_var = "floor_radius"
        if crater.nrings > 0:
            for i, ring in enumerate(crater.rings):
                pre = ring_prefix(i)
                x = ring.radius
                std_box(
                    var=f"{pre}_radius",
                    rmask=None,
                    color=colors[i + 5],
                    style="vertical",
                )
                std_box(
                    var=f"{pre}_rim_height",
                    rmask=rvals >= ring.radius,
                    color=colors[i + 5],
                    style="horizontal",
                    yoffset=crater.ring_offset(i),
                )
                ax.text(
                    rmax,
                    crater.ring_offset(i) + ring.rim_height,
                    f" {pre.title()}",
                    va="center",
                    ha="left",
                    fontsize=7,
                    color=colors[i + 5],
                )

                floor_var = f"{pre}_floor_radius"

        std_box(var=floor_var, rmask=None, color=colors[4], style="vertical")
        std_box(
            var="floor_elevation", rmask=rvals > 0, color=colors[4], style="horizontal"
        )
        ax.text(
            rmax,
            crater.floor_elevation,
            " Floor",
            va="center",
            ha="left",
            fontsize=7,
            color=colors[4],
        )

    if morphometric_vars is not None:
        # Add rings in the right order
        varlist = valid_fit_vars.copy()
        if crater.nrings > 0:
            for i in range(crater.nrings):
                pre = ring_prefix(i)
                varlist.extend([f"{pre}_{v}" for v in valid_fit_vars])

        fit_text = []
        for k in varlist:
            if k in morphometric_vars:
                if k in popt_map:
                    v = popt_map[k]
                elif hasattr(crater, k):
                    v = getattr(crater, k)
                else:
                    continue

                vstr = f"{k}: {v:.2f}"

                if psig_map is not None and k in psig_map:
                    v_std = psig_map[k]
                elif hasattr(crater, f"{k}_std"):
                    v_std = getattr(crater, f"{k}_std")
                else:
                    fit_text.apend(vstr)
                    continue
                if v_std > 1e-10:
                    vstr += f" ± {v_std:.3f}"
                fit_text.append(vstr)

        if len(fit_text) > 0:
            text = "\n".join(fit_text)
            ax.annotate(
                text=text,
                xy=(1.00, 0.10),
                xycoords="axes fraction",
                va="bottom",
                ha="left",
                fontsize=rcParams["font.size"] - 4,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 1.0,
                    "edgecolor": "black",
                },
                zorder=10000,
            )

    ax.legend(loc="upper left", fontsize=rcParams["font.size"] - 2)
    if filename is not None:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()
    plt.close()


def binned_profile(r: np.ndarray, h: np.ndarray, nrbins: int = 1000):
    """
    Compute a radially binned elevation profile.

    Parameters
    ----------
    r : numpy.ndarray
        Radial distances (m).
    h : numpy.ndarray
        Elevation values (m) at each radial distance.
    nrbins : int, optional
        Number of radial bins used for averaging. Default is 1000.

    Returns
    -------
    rbin : numpy.ndarray
        Midpoint radius for each non-empty radial bin.
    hbin : numpy.ndarray
        Mean elevation in each corresponding radial bin.
    hstd : numpy.ndarray
        Standard deviation of the data

    Notes
    -----
    Binning reduces overweighting of densely sampled rim points relative to floor
    points during profile fitting.
    """
    # Bin the hvals by radius so that we don't over-weight the more numerous rim points in the fit relative to the floor
    isort = np.argsort(r)
    df = pd.DataFrame({"r": r[isort], "h": h[isort]})
    hstd = np.full(nrbins, np.nan)
    nbins = nrbins + 1
    while any(np.isnan(hstd)) or any(hstd < 1e-9):
        nbins -= 1
        hvals = df.groupby(pd.cut(df["r"], bins=nbins), observed=True)["h"]
        hbin = hvals.mean()
        hstd = hvals.std()
        rbin = np.array([i.mid for i in hbin.index])
        if nbins < 10:
            raise RuntimeError(
                "Could not bin the profile. Likely due to not enough data."
            )
    return rbin, hbin.to_numpy(), hstd.to_numpy()


def get_profile_from_vectorfile(
    sim: Simulation, crater: BasicMoonCrater, feature: str, wrap: bool = False
):
    """
    Get the radial profile of the specified feature (rim, floor, or rim_flank) from vectorfiles.

    Files should be organized in folders by feature, and named {crater.label}.gpkg. i.e. for crater Copernicu's rim outline:
        data / vector_profiles / rim / Copernicus.gpkg

    Parameters
    ----------
    sim : Simulation
        The Simulation object containing the surface and morphology.
    crater : Crater
        The Crater object for which to get the profile.
    feature : str
        The feature for which to get the profile. Must be one of "rim", "floor", or "rim_flank".
    wrap : bool, optional
        If true, the profile will be repeated in in the -2pi to 0 and 2pi to 4pi theta values

    Returns
    -------
    rdata : np.ndarray
        The radial distances of the profile points from the crater center, normalized by the crater radius.
    bdata : np.ndarray
        The bearing angle in radians for each point.
    """
    import geopandas as gpd

    filename = vectorfiledir / feature / f"{crater.label}.gpkg"
    if not filename.exists():
        raise FileNotFoundError(f"{filename} not found")

    def cart2pol(x, y):
        # Assumes north-clockwise convention (0 is the y axis, positive is clockwise)
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(x, y)
        isort = np.argsort(theta)
        r = r[isort]
        theta = theta[isort]
        return r, theta

    gdf = (
        gpd.read_file(filename)
        .set_crs(sim.surface.crs, allow_override=True)
        .to_crs(sim.surface.local.crs)
    )

    xx, yy = gdf.geometry[0].exterior.coords.xy
    rdata, bdata = cart2pol(xx, yy)
    if wrap:
        rdata = np.hstack((rdata, rdata, rdata))
        bdata = np.hstack((bdata - 2 * np.pi, bdata, bdata + 2 * np.pi))
    return rdata, bdata


def save_morphometric_parameters(
    filename: Path,
    crater: BasicMoonCrater,
    popt_map: dict[str, float] = {},
    psig_map: dict[str, float] | None = None,
    isnew: bool = True,
    additional_data: dict = {},
    save_vars: list[str] | None = None,
):
    """
    Save the fitted morphometric parameters and their uncertainties to a CSV file.

    Parameters
    ----------
    filename : Path
        Output file path for the morphometric parameters CSV.
    crater : cratermaker.Crater
        Crater object corresponding to the fitted parameters.
    popt_map : dict
        A dict of fit variables as keys and the best-fit parameters from ``scipy.optimize.curve_fit`` as values
    psig_map : dict
        A dict of fit variables as keys and the standard error (sqrt of the diagonal of the covariance matrix) as avalues.
    isnew : bool
        If ``True``, a new CSV file is created and a header row is written. If ``False``, the parameters are appended to the existing file without writing a header.
    additional_data : dict, optional
        Optional dictionary of additional data to include in the CSV. Keys are column names and values are the corresponding data values for the current crater. These will be included as additional columns in the CSV after the standard parameter columns.
    save_vars : list[str], optional
        A list of morphometric variables to save. If None, then this will be constructed using the keys of the popt_map as well as a set of required variables for consistency in output.
    """
    if isnew:
        OpenTextMode = "w"
    else:
        OpenTextMode = "a"

    # Minimum additional data to save
    additional_data = {
        "longitude": crater.location[0],
        "latitude": crater.location[1],
        **additional_data,
    }

    if crater.nrings > 0:
        popt_map, psig_map, ring_popt_maps, ring_psig_maps = get_ring_popt_maps(
            list(popt_map.keys()), list(popt_map.values()), list(psig_map.values())
        )

    # These are the parameters that must appear in the file, ordered for consistency.

    ringskip_var = [
        "floor_elevation",
        "floor_blend",
        "frac_ejrim",
        "ejprofile",
        "peak_height",
        "peak_width",
        "peak_ring_radius",
        "latitude",
        "longitude",
        "dem_file_list",
        "D/pix",
    ]
    craterskip_var = ["parent"]
    if save_vars is None:
        save_vars = list(dict.fromkeys(list(popt_map.keys()) + ["id", "parent"]))
        if crater.nrings > 0 and ring_popt_maps is not None:
            for ring_popt_map in ring_popt_maps:
                save_vars = list(dict.fromkeys(save_vars + list(ring_popt_map.keys())))

        required_vars = [
            "radius",
            "floor_elevation",
            "rim_height",
            "rim_width",
            "floor_radius",
            "floor_blend",
            "wall_curvature",
            "frac_ejrim",
            "ejprofile",
            "peak_height",
            "peak_width",
            "peak_ring_radius",
        ]
        save_vars = list(dict.fromkeys(required_vars + save_vars))

    # This is done so that any rings get added as additional rows in the file after the main crater row
    popt_maps = [popt_map]
    psig_maps = [psig_map]
    craters = [crater]
    if crater.nrings > 0 and ring_popt_maps is not None:
        popt_maps.extend(ring_popt_maps)
        psig_maps.extend(ring_psig_maps)
        craters.extend(crater.rings)

    with Path(filename).open(OpenTextMode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if isnew:
            header = ["name", "label", "morphology_type"]

            if popt_map is not None:
                for k in save_vars:
                    header.append(k)
                    if psig_map is not None and (k in psig_map or k in required_vars):
                        header.append(f"{k}_std")
            if additional_data is not None:
                for k in additional_data.keys():
                    header.append(k)
            writer.writerow(header)
        for c, popt, psig in zip(craters, popt_maps, psig_maps, strict=True):
            row = [c.name, c.label, c.morphology_type]
            for k in save_vars:
                if (c.morphology_type == "ring" and k in ringskip_var) or (
                    c.morphology_type != "ring" and k in craterskip_var
                ):
                    v = ""
                else:
                    if k in popt:
                        v = popt.get(k)
                    elif hasattr(c, k):
                        v = getattr(c, k)
                    else:
                        raise ValueError(f"{k} not found in popt_map or crater object")
                row.append(v)
                if psig is not None and (k in psig or k in required_vars):
                    if c.morphology_type == "ring" and k in ringskip_var:
                        vsig = ""
                    else:
                        vsig = psig.get(k, 0)
                    row.append(vsig)
            if additional_data is not None:
                for k in additional_data:
                    if (c.morphology_type == "ring" and k in ringskip_var) or (
                        c.morphology_type != "ring" and k in craterskip_var
                    ):
                        row.append("")
                    else:
                        row.append(additional_data[k])
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit crater morphometry parameters to a local DTM.",
    )
    parser.add_argument(
        "-p",
        "--profile-style",
        type=str,
        default="combined",
        help="Profile preparation method before fitting. Currently 'combined', 'binned', and 'radial' are supported.",
    )
    parser.add_argument(
        "-i",
        "--input-csv",
        type=str,
        default=default_input_csv,
        help="Input CSV file with crater data.",
    )
    parser.add_argument(
        "-n", "--nrbins", type=int, default=500, help="Number of radial bins"
    )
    parser.add_argument(
        "-t",
        "--nthetabins",
        type=int,
        default=360,
        help="Number of radial slices when using the radial profile_style",
    )
    parser.add_argument(
        "-e",
        "--plot-extension",
        type=str,
        default="pdf",
        help="File extension for profile plots.",
    )
    parser.add_argument(
        "-r",
        "--redo-crater",
        type=str,
        default=None,
        help="Only perform analysis from a particular crater from the input file. New results will be appended to the output file.",
    )
    parser.add_argument(
        "-R",
        "--restart-crater",
        type=str,
        default=None,
        help="Restart analysis starting from a particular crater from the input file. New results will be appended to the output file.",
    )
    parser.add_argument(
        "-l",
        "--fit-range-lo",
        type=float,
        default=None,
        help="Lower bound of radius range to use for fitting relative to crater radius (e.g., 0.5 for 50%% of crater radius)",
    )
    parser.add_argument(
        "-u",
        "--fit-range-hi",
        type=float,
        default=1.5,
        help="Upper bound of radius range to use for fitting relative to crater radius (e.g., 2.0 for 200%% of crater radius)",
    )
    parser.add_argument(
        "-D",
        "--fit-rim",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If set, radius will be included in the fit, but will be determined from the data.",
    )
    parser.add_argument(
        "-o",
        "--save-intermediate",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, the intermediate results will be plotted for each fit.",
    )
    parser.add_argument(
        "--plot-map",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If true, make draw the rim and floor values on a map of elevation (only for radial style).",
    )
    args = parser.parse_args()

    profile_style = args.profile_style
    input_csv = args.input_csv
    nrbins = args.nrbins
    nthetabins = args.nthetabins
    plot_extension = args.plot_extension
    redo_crater = args.redo_crater
    restart_crater = args.restart_crater
    fit_range = (args.fit_range_lo, args.fit_range_hi)
    fit_rim = args.fit_rim
    save_intermediate = args.save_intermediate
    plot_map = args.plot_map

    craters = get_input_craters(input_csv)

    if restart_crater is not None:
        redo_crater = restart_crater

    if redo_crater is None:
        isnew = True
    else:
        isnew = False
    for i, crater in enumerate(craters):
        if redo_crater is not None and crater.label != redo_crater:
            continue
        if restart_crater is not None:
            redo_crater = None
            restart_crater = None
        print(f"{i + 1} of {len(craters)}: Processing {crater.name} ({crater.label})")

        dem_file_list = (
            crater.dem_file_list if type(crater.dem_file_list) is list else None
        )

        print("Getting Simulation object...")
        sim = get_sim(crater=crater, name_append=None)
        dem_file_list = [f"'{v}'" for v in sim.surface.dem_file_list]
        dem_file_list = ";".join(dem_file_list)

        additional_data = {"dem_file_list": dem_file_list}

        if profile_style not in ["radial", "binned", "combined", "psd"]:
            raise ValueError(f"{profile_style} is not a valid profile style.")
        ignore_vars = ["peak_center_distance", "peak_center_bearing", "floor_elevation"]

        if profile_style in ["radial", "combined"]:
            print(f"Getting morphometry for {crater.label} from radial slices.")
            fit_vars = [v for v in valid_fit_vars if v not in ignore_vars]
            popt_map, psig_map = get_morphometry_from_radial_slices(
                sim=sim,
                crater=crater,
                fit_vars=fit_vars,
                nrbins=nrbins,
                nthetabins=nthetabins,
                fit_range=fit_range,
                plot_extension=plot_extension,
                save_intermediate=save_intermediate,
                fit_rim=fit_rim,
                plot_map=plot_map,
            )
            # Update the crater with new values for a combined profile
            location = popt_map.pop("location", crater.location)
            _ = psig_map.pop("location")
            crater = DummyCrater(
                crater=crater, popt_map=popt_map, psig_map=psig_map, location=location
            )
            print("Saving morphometric parameters to CSV...")

            save_morphometric_parameters(
                filename=datadir / "minton_morphometry_radial.csv",
                crater=crater,
                popt_map=popt_map,
                psig_map=psig_map,
                isnew=isnew,
                additional_data=additional_data,
            )

        if profile_style in ["binned", "combined"]:
            print(f"Getting morphometry for {crater.label} from binned profile.")

            fit_vars = [v for v in valid_fit_vars if v not in ignore_vars]
            popt_map, psig_map = get_morphometry_from_binned_profile(
                sim=sim,
                crater=crater,
                fit_vars=fit_vars,
                nrbins=nrbins,
                fit_range=fit_range,
                plot_extension="png",
                save_intermediate=save_intermediate,
                fit_rim=fit_rim,
            )
            crater.rings = []
            crater = DummyCrater(crater=crater, popt_map=popt_map, psig_map=psig_map)
            print("Saving morphometric parameters to CSV...")
            additional_data["D/pix"] = crater.diameter / sim.surface.pix
            save_morphometric_parameters(
                filename=datadir / "minton_morphometry_binned.csv",
                crater=crater,
                popt_map=popt_map,
                psig_map=psig_map,
                isnew=isnew,
                additional_data=additional_data,
            )

        isnew = False

        del sim
