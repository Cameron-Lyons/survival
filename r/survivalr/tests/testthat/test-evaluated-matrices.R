for (.evaluated_kind in c("coxph", "survreg")) for (.evaluated_rhs in .evaluated_matrix_setup()$formulas) {
  if (.evaluated_kind == "survreg" && grepl("frailty", .evaluated_rhs)) next
  test_that(paste("evaluated frames match stock", .evaluated_kind, .evaluated_rhs), {
  kind <- .evaluated_kind
  rhs <- .evaluated_rhs
  original <- options(na.action = "na.omit")
  on.exit(options(original), add = TRUE)
  setup <- .evaluated_matrix_setup()
    for (named in c(FALSE, TRUE)) for (cached in c(FALSE, TRUE)) {
      options(na.action = "na.omit")
      d <- .aft_matrix_input_data(setup, "complete", named)
      form <- as.formula(paste("Surv(futime,fustat)~", rhs))
      fit <- function(namespace) get(kind, asNamespace(namespace))(
        form, d, x = cached, model = TRUE)
      actual <- .logical_matrix_capture(function() fit("survivalr"))
      stock <- .logical_matrix_capture(function() fit("survival"))
      expect_identical(gsub("[[:space:]]+", "", actual$warnings),
        gsub("[[:space:]]+", "", stock$warnings))
      source <- model.frame(stock$value)
      saved <- list(coef = coef(actual$value), variance = vcov(actual$value),
        matrix = model.matrix(actual$value))
      for (action in c("na.omit", "na.exclude", "na.pass", "na.fail")) {
        options(na.action = action)
        for (input in setup$inputs) {
          nd <- .evaluated_matrix_newdata(source, input)
          marker <- attr(nd, "terms")
          columns <- nd
          attr(columns, "terms") <- NULL
          before <- serialize(columns, NULL)
          result <- .logical_matrix_capture(function() model.matrix(actual$value, nd))
          expected <- .logical_matrix_capture(function() model.matrix(stock$value, nd))
          info <- paste(kind, rhs, named, cached, action, input)
          if (is.list(expected$value) && !is.null(expected$value$error)) {
            expect_identical(length(result$value$error), 1L, info = info)
            # Python missing-column errors carry the column name and cause.
            cause <- if (grepl("not found", expected$value$error) ||
              expected$value$error == "number of variables != number of variable names")
              "not found" else expected$value$error
            if (kind == "coxph" && grepl("strata(", rhs, fixed = TRUE) &&
              !grepl("strata\\([^)]*\\):", rhs) &&
              grepl("object '[gh]' not found", expected$value$error))
              cause <- "must contain the strata"
            expect_match(result$value$error, cause, fixed = TRUE, info = info)
          } else {
            .logical_matrix_compare(result$value, expected$value, info)
            expect_identical(is.na(result$value), is.na(expected$value), info = info)
            expect_identical(is.nan(result$value), is.nan(expected$value), info = info)
          }
          expect_identical(gsub("[[:space:]]+", "", .formula_warning_messages(result$warnings)),
            gsub("[[:space:]]+", "", expected$warnings), info = info)
          columns <- nd
          attr(columns, "terms") <- NULL
          expect_identical(serialize(columns, NULL), before, info = info)
          expect_identical(attr(nd, "terms"), marker, info = info)
        }
      }
      expect_identical(list(coef = coef(actual$value), variance = vcov(actual$value),
        matrix = model.matrix(actual$value)), saved)
    }
  })
}
