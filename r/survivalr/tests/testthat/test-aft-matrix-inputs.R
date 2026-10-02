test_that("AFT matrices evaluate covariates after removing standalone strata", {
  setup <- .aft_matrix_input_setup()
  for (rhs in setup$formulas) for (variant in c("complete", "missing", "subset"))
    for (named in c(FALSE, TRUE)) for (cached in c(FALSE, TRUE)) {
      data <- .aft_matrix_input_data(setup, variant, named)
      info <- paste(rhs, variant, named, cached)
      stock <- .logical_matrix_capture(function() .aft_matrix_input_fit(rhs,data,variant,cached,"survival"))
      actual <- .logical_matrix_capture(function() .aft_matrix_input_fit(rhs,data,variant,cached,"survivalr"))
      expect_identical(gsub("[[:space:]]+", "", actual$warnings),
                       gsub("[[:space:]]+", "", stock$warnings), info = info)
      for (input in setup$inputs) {
        nd <- .aft_matrix_input_newdata(data, input)
        call <- function(fit) if(is.null(nd)) model.matrix(fit) else model.matrix(fit, data=nd)
        expected <- .logical_matrix_capture(function() call(stock$value))
        result <- .logical_matrix_capture(function() call(actual$value))
        if(is.list(expected$value) && !is.null(expected$value$error)) {
          # Failures compare their cause. Stock also warns before rejecting a
          # numeric replacement for an ordinary factor; the port rejects it directly.
          if (grepl("not found", expected$value$error)) {
            column <- sub("object '([^']+)'.*", "\\1", expected$value$error)
            expect_match(result$value$error, paste0("column '", column, "' not found"), info = paste(info,input))
          } else {
            expect_match(result$value$error, "unknown level|new level|contrasts apply only", info = paste(info,input))
          }
        } else {
          expect_identical(sub(" in log\\(off\\)$", "", result$warnings),
                           expected$warnings, info = paste(info,input))
          .logical_matrix_compare(result$value, expected$value, paste(info,input))
        }
      }
    }
})

test_that("AFT prediction still evaluates its scale strata and offset", {
  data <- .aft_matrix_input_setup()$data
  actual <- survreg(Surv(futime,fustat) ~ age + strata(g) + offset(off), data, x = TRUE)
  stock <- survival::survreg(Surv(futime,fustat) ~ age + strata(g) + offset(off), data, x = TRUE)
  nd <- data[c(7L,3L,7L,1L),]
  nd$g[2L] <- NA
  expect_equal(predict(actual, nd, type = "uquantile", p = c(.2,.8), na.action = na.omit),
               predict(stock, nd, type = "uquantile", p = c(.2,.8), na.action = na.omit) + nd$off[-2L],
               tolerance = 3e-7)
  nd$g <- NULL
  expect_error(predict(actual, nd), "column 'g' not found")
  expect_equal(model.matrix(actual, nd), model.matrix(stock, nd))
})
