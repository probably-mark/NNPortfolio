## -----------------------------------
# Custom layer class
## -----------------------------------
Portfolio_Layer <- R6::R6Class("KerasLayer",
                               inherit = KerasLayer,
                               public = list(
                                 output_dim = NULL,
                                 kernel = NULL,
                                 # pool_size = NULL,
                                 initialize = function(output_dim) {
                                   self$output_dim <- output_dim
                                 },
                                 call = function(input_tensor, mask = NULL) {
                                   # we want to set all but top `portfolio_size to zero
                                   n <- as.numeric(k_int_shape(input_tensor)[-1])
                                   if ((portfolio_size + 1) <= n) {
                                     # sets our first element (the smallest in each row) to zero
                                     x <- input_tensor - k_min(input_tensor, 2, keepdims = TRUE)
                                     # a will be our output tensor
                                     a <- x
                                     # Number of iterations to end with non-zeroed portfolio size
                                     M <- n - portfolio_size - 1
                                     # for loop to set remaining to zero
                                     for (i in 1:M) {
                                       # turn 0 -> 1
                                       a <- a + 1
                                       # create matrix of ones of same size of a
                                       y <- k_ones_like(a)
                                       # boolean matrix when a = y
                                       z <- k_equal(a,y)
                                       # cast boolean matrix to tensor
                                       z <- tf$cast(z, dtype="float32") # *x seems to be required for it to return 1 and 0's correctly
                                       # add boolean matrix
                                       a <- a + z
                                       # now min will capture the second smallest value in the row
                                       row_min <- k_min(a, 2, keepdims = TRUE)
                                       # the previous min will be negative
                                       a <- a - z - row_min
                                       # # convert negative to 0
                                       a <- k_clip(a, 0, 1)
                                       # find column sums for normalization
                                       a_sum <- k_sum(a, axis = 2, keepdims = TRUE)
                                       # convert to weights that sum to 1
                                       a <- a/a_sum
                                     }
                                     a <- tf$cast(a, dtype="float32")
                                     a
                                   }
                                   else {
                                     a <- input_tensor
                                     a <- tf$cast(a, dtype="float32")
                                     a
                                   }
                                 },
                                 compute_output_shape = function(input_shape) {
                                   list(input_shape[[1]], self$output_dim)
                                 }
                               )
)
## -----------------------------------
# Create layer wrapper function
## -----------------------------------
layer_portfolio <- function(object, output_dim, name = NULL) {
  create_layer(Portfolio_Layer, object, list(
    output_dim = as.integer(output_dim),
    name = name
  ))
}

## -----------------------------------
# prepare fixed weights for final layer
# to add weighted vectors together
# without additional NN_weights
## -----------------------------------
constant_weights <- function(shape) {
  initial <- tf$constant(1, shape=shape)
  tf$Variable(initial)
}

## -----------------------------------
# Define Loss function that equates
# to solving the maximum Sharpe Ratio
## -----------------------------------
Sharpe_Loss <- function(y_true, y_pred) {
  K <- backend()
  # Equation 2.20 in Campbell & Viceira (2006): Power utility with CRRA=gamma
  loss <- - (K$log(K$mean(1 + y_pred)) - 1/2*gamma*K$var(y_pred))
  # loss <- K$mean(K$square(y_true - y_pred))/100
  loss
}

## -----------------------------------
# Check the actual Sharpe ratio
## -----------------------------------
Sharpe_Metric <- function(y_true, y_pred) {
  K <- backend()
  # Equation 2.20 in Campbell & Viceira (2006): Power utility with CRRA=gamma
  # loss <- - (K$log((K$mean(1 + y_pred)) - 1/2*gamma*K$std(y_true)))
  metric <- K$mean(y_pred)/K$std(y_pred)
  metric
}
attr(Sharpe_Metric, "py_function_name") <- "Sharpe_Metric"

## -----------------------------------
# Embed it all into one function
## -----------------------------------
NN_Portfolio <- function(data, portfolio_size = 30, gamma = 2, validation_percent = 0.2, epochs = 5, seed = 123) {
  cat('\nPort Size = ', portfolio_size)
  N <- dim(data)[2]
  cat('\nN = ', N, '\n')
  # use_session_with_seed(123, quiet = TRUE)

  # Custom layer class
  Portfolio_Layer <- R6::R6Class("KerasLayer",
                           inherit = KerasLayer,
                           public = list(
                             output_dim = NULL,
                             kernel = NULL,
                             # pool_size = NULL,
                             initialize = function(output_dim) {
                               self$output_dim <- output_dim
                             },
                             call = function(input_tensor, mask = NULL) {
                               # we want to set all but top `portfolio_size to zero
                               n <- as.numeric(k_int_shape(input_tensor)[-1])
                               if ((portfolio_size + 1) <= n) {
                                 # sets our first element (the smallest in each row) to zero
                                 x <- input_tensor - k_min(input_tensor, 2, keepdims = TRUE)
                                 # a will be our output tensor
                                 a <- x
                                 # Number of iterations to end with non-zeroed portfolio size
                                 M <- n - portfolio_size - 1
                                 # for loop to set remaining to zero
                                 for (i in 1:M) {
                                   # turn 0 -> 1
                                   a <- a + 1
                                   # create matrix of ones of same size of a
                                   y <- k_ones_like(a)
                                   # boolean matrix when a = y
                                   z <- k_equal(a,y)
                                   # cast boolean matrix to tensor
                                   z <- tf$cast(z, dtype="float32") # *x seems to be required for it to return 1 and 0's correctly
                                   # add boolean matrix
                                   a <- a + z
                                   # now min will capture the second smallest value in the row
                                   row_min <- k_min(a, 2, keepdims = TRUE)
                                   # the previous min will be negative
                                   a <- a - z - row_min
                                   # # convert negative to 0
                                   a <- k_clip(a, 0, 1)
                                   # find column sums for normalization
                                   a_sum <- k_sum(a, axis = 2, keepdims = TRUE)
                                   # convert to weights that sum to 1
                                   a <- a/a_sum
                                 }
                                 a <- tf$cast(a, dtype="float32")
                                 a
                               }
                               else {
                                 a <- input_tensor
                                 a <- tf$cast(a, dtype="float32")
                                 a
                               }
                             },
                             compute_output_shape = function(input_shape) {
                               list(input_shape[[1]], self$output_dim)
                             }
                           )
  )
  # Create layer wrapper function
  layer_portfolio <- function(object, output_dim, name = NULL) {
    create_layer(Portfolio_Layer, object, list(
      output_dim = as.integer(output_dim),
      name = name
    ))
  }
  # attr(New_Layer, "py_function_name") <- "New_Layer"
  # attr(layer_new, "py_function_name") <- "layer_new"

  # prepare fixed weights for final layer
  # to add weighted vectors together
  # without additional NN_weights
  constant_weights <- function(shape) {
    initial <- tf$constant(1, shape=shape)
    tf$Variable(initial)
  }

  # Define Loss function that equates
  # to solving the maximum Sharpe Ratio
  Sharpe_Loss <- function(y_true, y_pred) {
    K <- backend()
    # Equation 2.20 in Campbell & Viceira (2006): Power utility with CRRA=gamma
    loss <- - (K$log(K$mean(1 + y_pred)) - 1/2*gamma*K$var(y_pred))
    # loss <- K$mean(K$square(y_true - y_pred))/100
    loss
  }

  # Check the actual Sharpe ratio
  Sharpe_Metric <- function(y_true, y_pred) {
    K <- backend()
    # Equation 2.20 in Campbell & Viceira (2006): Power utility with CRRA=gamma
    # loss <- - (K$log((K$mean(1 + y_pred)) - 1/2*gamma*K$std(y_true)))
    metric <- K$mean(y_pred)/K$std(y_pred)
    metric
  }
  attr(Sharpe_Metric, "py_function_name") <- "Sharpe_Metric"

  input <- layer_input(shape = N, name = "Input")
  LSTM <- layer_dense(input, units = N, name = "LSTM", kernel_initializer = initializer_glorot_uniform(seed = seed))
  # Relu <- layer_dense(input, units = N, activation = "relu", name = "Plain", kernel_initializer = initializer_random_uniform(minval = -1, maxval = 1, seed = seed))
  Sweights <- layer_dense(LSTM, units = N, activation = "softmax", name = "Sweights", kernel_initializer = initializer_random_uniform(minval = 0, maxval = 1, seed = seed+1)) # , dtype = tf$float32
  Portfolio_Layer <- layer_portfolio(Sweights, output_dim = N, name = "Pweights")
  Multiply <- layer_multiply(list(input, New_Layer), name = "Multiply")
  Sum <- layer_dense(Multiply, units = 1, name = "Sum", use_bias = FALSE, kernel_initializer = initializer_constant(1.0), trainable = FALSE)
  mymodel <- keras_model(input, Sum)
  summary(mymodel)

  mymodel %>%
    compile(
      optimizer = optimizer_rmsprop(lr = 0.001), # adam or adagrad
      loss = New_Loss, # New_Loss
      metrics = New_Metric
    )

  mymodel %>%
    fit(
      x = data, y = data[,1], # y is not used, so just pass anything in
      epochs = epochs,
      validation_split = validation_percent,
      verbose = 2,
      shuffle = FALSE
    )

  # ---- Save results in a list
  ret <- structure(list(model = mymodel),
                   class = "NN_Portfolio")
  return(invisible(ret))
}


