#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Altrep.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>

/* An opaque handle: serialization invokes an R closure while the Python
 * object is live, and stores its state bundle. Fits allocate no serialized copy.
 * After loading, data2 holds that bundle until R restores the Python object. */
static R_altrep_class_t snapshot_class;

static void check_snapshot(SEXP value) {
    if (!ALTREP(value) || !R_altrep_inherits(value, snapshot_class))
        Rf_error("Invalid survival model serialization state");
}

static SEXP snapshot_state(SEXP value) {
    SEXP callback = R_altrep_data1(value);
    if (callback == R_NilValue)
        return R_altrep_data2(value);
    SEXP call = PROTECT(Rf_lang1(callback));
    SEXP state = PROTECT(Rf_eval(call, R_GlobalEnv));
    if (TYPEOF(state) != VECSXP || XLENGTH(state) == 0)
        Rf_error("Survival model serialization must return a state bundle");
    UNPROTECT(2);
    return state;
}

static SEXP snapshot_restore(SEXP class_info, SEXP state) {
    (void) class_info;
    if (TYPEOF(state) != VECSXP || XLENGTH(state) == 0)
        Rf_error("Invalid serialized survival model state");
    return R_new_altrep(snapshot_class, R_NilValue, state);
}

static SEXP snapshot_duplicate(SEXP value, Rboolean deep) {
    (void) deep;
    return R_new_altrep(snapshot_class, R_altrep_data1(value), R_altrep_data2(value));
}

static R_xlen_t snapshot_length(SEXP value) { (void) value; return 1; }
static Rbyte snapshot_element(SEXP value, R_xlen_t index) {
    (void) value; (void) index; return 0;
}
static const void *snapshot_pointer_or_null(SEXP value) { (void) value; return NULL; }

static void *snapshot_pointer(SEXP value, Rboolean writable) {
    (void) value; (void) writable;
    /* Serialization version 2 materializes ALTREP and would discard state. */
    Rf_error("Survival model persistence requires serialization version = 3");
    return NULL;
}

static SEXP snapshot_new(SEXP callback) {
    if (!Rf_isFunction(callback))
        Rf_error("Survival model snapshot callback must be a function");
    return R_new_altrep(snapshot_class, callback, R_NilValue);
}

static SEXP snapshot_get(SEXP value) {
    check_snapshot(value);
    return snapshot_state(value);
}

static SEXP snapshot_bind(SEXP value, SEXP callback) {
    check_snapshot(value);
    if (!Rf_isFunction(callback))
        Rf_error("Survival model snapshot callback must be a function");
    R_set_altrep_data1(value, callback);
    R_set_altrep_data2(value, R_NilValue);
    return R_NilValue;
}

static const R_CallMethodDef call_methods[] = {
    {"snapshot_new", (DL_FUNC) &snapshot_new, 1},
    {"snapshot_get", (DL_FUNC) &snapshot_get, 1},
    {"snapshot_bind", (DL_FUNC) &snapshot_bind, 2},
    {NULL, NULL, 0}
};

void attribute_visible R_init_survivalr(DllInfo *dll) {
    R_registerRoutines(dll, NULL, call_methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
    snapshot_class = R_make_altraw_class("python_model_state", "survivalr", dll);
    R_set_altrep_Serialized_state_method(snapshot_class, snapshot_state);
    R_set_altrep_Unserialize_method(snapshot_class, snapshot_restore);
    R_set_altrep_Duplicate_method(snapshot_class, snapshot_duplicate);
    R_set_altrep_Length_method(snapshot_class, snapshot_length);
    R_set_altvec_Dataptr_method(snapshot_class, snapshot_pointer);
    R_set_altvec_Dataptr_or_null_method(snapshot_class, snapshot_pointer_or_null);
    R_set_altraw_Elt_method(snapshot_class, snapshot_element);
}
