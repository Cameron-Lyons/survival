"""R survival's data sets as dictionaries of columns.

Factor columns are :class:`~survival.r.Factor` lists that carry R's level order,
so models fitted on them use R's reference levels.
"""

from functools import wraps

from ._binding_utils import bind_names
from ._factor import Factor

__all__ = bind_names(
    globals(),
    [
        "load_aml",
        "load_bladder",
        "load_bladder1",
        "load_bladder2",
        "load_braking",
        "load_capacitor",
        "load_cgd",
        "load_cgd0",
        "load_colon",
        "load_cracks",
        "load_diabetic",
        "load_flchain",
        "load_gbsg",
        "load_genfan",
        "load_heart",
        "load_hoel",
        "load_ifluid",
        "load_imotor",
        "load_jasa",
        "load_jasa1",
        "load_kidney",
        "load_logan",
        "load_lung",
        "load_mgus",
        "load_mgus1",
        "load_mgus2",
        "load_myeloid",
        "load_myeloma",
        "load_nafld1",
        "load_nafld2",
        "load_nafld3",
        "load_nwtco",
        "load_ovarian",
        "load_pbc",
        "load_pbcseq",
        "load_rats",
        "load_rats2",
        "load_retinopathy",
        "load_rhdnase",
        "load_rotterdam",
        "load_solder",
        "load_stanford2",
        "load_tobin",
        "load_transplant",
        "load_turbine",
        "load_udca",
        "load_udca1",
        "load_udca2",
        "load_valveseat",
        "load_veteran",
        "load_cancer",
        "load_leukemia",
    ],
)

_FACTOR_LEVELS: dict[str, dict[str, tuple[str, ...]]] = {
    "aml": {
        "x": ("Maintained", "Nonmaintained"),
    },
    "bladder1": {
        "treatment": ("placebo", "pyridoxine", "thiotepa"),
    },
    "cgd": {
        "center": (
            "Harvard Medical Sch",
            "Scripps Institute",
            "Copenhagen",
            "NIH",
            "L.A. Children's Hosp",
            "Mott Children's Hosp",
            "Univ. of Utah",
            "Univ. of Washington",
            "Univ. of Minnesota",
            "Univ. of Zurich",
            "Texas Children's Hosp",
            "Amsterdam",
            "Mt. Sinai Medical Ctr",
        ),
        "treat": ("placebo", "rIFN-g"),
        "sex": ("male", "female"),
        "inherit": ("X-linked", "autosomal"),
        "hos.cat": ("US:NIH", "US:other", "Europe:Amsterdam", "Europe:other"),
    },
    "colon": {
        "rx": ("Obs", "Lev", "Lev+5FU"),
    },
    "diabetic": {
        "laser": ("xenon", "argon"),
        "eye": ("left", "right"),
    },
    "flchain": {
        "sex": ("F", "M"),
        "chapter": (
            "Blood",
            "Circulatory",
            "Congenital",
            "Digestive",
            "Endocrine",
            "External Causes",
            "Genitourinary",
            "Ill Defined",
            "Infectious",
            "Injury and Poisoning",
            "Mental",
            "Musculoskeletal",
            "Neoplasms",
            "Nervous",
            "Respiratory",
            "Skin",
        ),
    },
    "heart": {
        "transplant": ("0", "1"),
    },
    "hoel": {
        "trt": ("Control", "Germ-free"),
        "outcome": ("censor", "thymic lymphoma", "reticulum cell sarcoma", "other"),
    },
    "kidney": {
        "disease": ("Other", "GN", "AN", "PKD"),
    },
    "leukemia": {
        "x": ("Maintained", "Nonmaintained"),
    },
    "logan": {
        "occupation": ("farm", "operatives", "craftsmen", "sales", "professional"),
        "focc": ("farm", "operatives", "craftsmen", "sales", "professional"),
        "race": ("non-black", "black"),
    },
    "mgus": {
        "sex": ("female", "male"),
        "pcdx": ("AM", "LP", "MA", "MM"),
    },
    "mgus1": {
        "sex": ("female", "male"),
        "pcdx": ("AM", "LP", "MA", "MM"),
        "event": ("censor", "pcm", "death"),
    },
    "mgus2": {
        "sex": ("F", "M"),
    },
    "myeloid": {
        "sex": ("f", "m"),
        "flt3": ("A", "B", "C"),
    },
    "nafld3": {
        "event": (
            "afib",
            "ang/isc",
            "cardiac arrest",
            "diabetes",
            "heart failure",
            "htn",
            "dyslipidemia",
            "MI",
            "nafld",
            "stroke",
        ),
    },
    "pbc": {
        "sex": ("m", "f"),
    },
    "pbcseq": {
        "sex": ("m", "f"),
    },
    "retinopathy": {
        "laser": ("xenon", "argon"),
        "eye": ("right", "left"),
        "type": ("juvenile", "adult"),
    },
    "rotterdam": {
        "size": ("<=20", "20-50", ">50"),
    },
    "solder": {
        "Opening": ("L", "M", "S"),
        "Solder": ("Thick", "Thin"),
        "Mask": ("A1.5", "A3", "A6", "B3", "B6"),
        "PadType": ("D4", "D6", "D7", "L4", "L6", "L7", "L8", "L9", "W4", "W9"),
        "Panel": ("1", "2", "3"),
    },
    "transplant": {
        "sex": ("m", "f"),
        "abo": ("A", "B", "AB", "O"),
        "event": ("censored", "death", "ltx", "withdraw"),
    },
    "veteran": {
        "celltype": ("squamous", "smallcell", "adeno", "large"),
    },
}


def _with_factor_levels(loader, levels):
    @wraps(loader)
    def load():
        data = loader()
        for column, categories in levels.items():
            data[column] = Factor(data[column], categories)
        return data

    return load


_PYTHON_WRAPPERS = frozenset(f"load_{dataset}" for dataset in _FACTOR_LEVELS)
for _name in _PYTHON_WRAPPERS:
    globals()[_name] = _with_factor_levels(globals()[_name], _FACTOR_LEVELS[_name[5:]])
del _name
