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

test_that("response and matrix AFT fits retain design names through missing rows", {
  d <- survival::ovarian
  rownames(d) <- paste0("patient / ", rev(seq_len(nrow(d))))
  x <- as.matrix(d[c("age", "rx")])
  actual <- survreg(Surv(d$futime, d$fustat), x = x)
  stock <- survival::survreg(survival::Surv(futime, fustat) ~ 0 + age + rx, d, x = TRUE)
  expect_equal(model.matrix(actual), model.matrix(stock))

  nd <- x[c(5L, 7L, 19L, 25L), , drop = FALSE]
  rownames(nd) <- c("first", "二", "three", "00:04")
  nd[2L, "age"] <- NA_real_
  frame <- as.data.frame(nd)
  # Matrix predictions use fitted column order even without column names.
  colnames(nd) <- NULL
  expect_equal(model.matrix(actual, nd), model.matrix(stock, frame))
  for (type in c("terms", "link", "response", "quantile", "uquantile")) {
    expect_equal(predict(actual, type = type, se.fit = TRUE),
                 predict(stock, type = type, se.fit = TRUE), tolerance = 3e-7)
    for (action in c("na.pass", "na.omit", "na.exclude")) {
      expect_equal(predict(actual, nd, type = type, se.fit = TRUE, na.action = action),
                   predict(stock, frame, type = type, se.fit = TRUE, na.action = action),
                   tolerance = 3e-7)
    }
  }
  nd <- as.matrix(frame)
  storage.mode(nd) <- "integer"
  expect_equal(model.matrix(actual, nd), model.matrix(stock, as.data.frame(nd)))
  expect_equal(predict(actual, nd, type = "terms", na.action = na.omit),
               predict(stock, as.data.frame(nd), type = "terms", na.action = na.omit),
               tolerance = 3e-7)
})
