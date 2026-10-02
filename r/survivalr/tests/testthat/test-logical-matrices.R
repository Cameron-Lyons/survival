test_that("logical matrices and all metadata follow stock R", {
  setup <- .logical_matrix_setup()
  for (kind in c("coxph", "survreg")) for (rhs in setup$formulas)
    for (variant in c("mixed", "false", "true", "missing", "subset")) for (named in c(FALSE, TRUE)) {
      data <- .logical_matrix_data(setup, variant, named)
      info <- paste(kind,rhs,variant,named)
      stock <- .logical_matrix_capture(function() .logical_matrix_fit(kind,rhs,data,variant,"survival"))
      actual <- .logical_matrix_capture(function() .logical_matrix_fit(kind,rhs,data,variant,"survivalr"))
      expect_identical(gsub("[[:space:]]+", "", actual$warnings),
                       gsub("[[:space:]]+", "", stock$warnings), info = info)
      if (is.list(stock$value) && !is.null(stock$value$error)) {
        expect_match(actual$value$error, "at least two levels|contrasts can be applied", info = info)
        next
      }
      expect_identical(names(coef(actual$value)), names(coef(stock$value)), info = info)
      expect_equal(as.numeric(coef(actual$value)), as.numeric(coef(stock$value)), tolerance = 3e-7, info = info)
      for (input in c("stored", "complete", "partial", "all_missing", "empty", "single")) {
        nd <- .logical_matrix_newdata(data, input)
        call <- function(fit) if(is.null(nd)) model.matrix(fit) else model.matrix(fit, data=nd)
        expected <- .logical_matrix_capture(function() call(stock$value))
        result <- .logical_matrix_capture(function() call(actual$value))
        expect_identical(result$warnings, expected$warnings, info = paste(info,input))
        if(is.list(expected$value) && !is.null(expected$value$error))
          expect_type(result$value$error, "character", info = paste(info,input)) else
          .logical_matrix_compare(result$value, expected$value, paste(info,input))
      }
    }
})
