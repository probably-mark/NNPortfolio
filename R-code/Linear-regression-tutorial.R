# -----------------------------------------------------------------------------
# Import Packages
# -----------------------------------------------------------------------------
library(tidyverse)
library(keras)
tensorflow::tf_config()
library(tensorflow)

# portfolio_size = 3; gamma = 2; validation_percent = 0; epochs = 5;
# set.seed(1)

# -----------------------------------------------------------------------------
# Prepare Data
# -----------------------------------------------------------------------------
beta_0 = 0
beta_1 = 1
beta_2 = 0.5

df <- data.frame(
        x1 = c(1,2,3,4),
        x2 = c(4,8,10,6)
      )

df <- df %>% mutate(y = beta_0 + beta_1 * x1 + beta_2 * x2)
x = as.matrix(df[,1:2]) # collect these into a matrix object for later
y = as.matrix(df[,3])   # collect these into a matrix object for later
df

# -----------------------------------------------------------------------------
# Linear Regression
# -----------------------------------------------------------------------------
m <- lm(y ~ x1 + x2, df)
m

# -----------------------------------------------------------------------------
# Neural Network
# -----------------------------------------------------------------------------
# Design the network connections
input <- layer_input(shape = 2, name = "Input")
linear_layer <- layer_dense(input, units = 1, name = "Linear", kernel_initializer = initializer_constant(1.0))
model_one <- keras_model(input, linear_layer)
model_one

# Set the hyper-parameters, loss and metric functions
model_one %>% 
    compile(
      optimizer = optimizer_rmsprop(learning_rate = 0.01), # stands for Root Mean Square Propagation
      loss = "mean_squared_error", # Use MSE to match linear regression
      metrics = "mean_squared_error"
    )

# Fit one iteration (called an epoch)
model_one %>% 
    fit(
      x = x, y = y,
      epochs = 1,
      validation_split = 0,
      verbose = 2,
      shuffle = FALSE
    )

layer_name <- 'Linear'
intermediate_layer_model <- keras_model(inputs = model_one$input,
                                        outputs = get_layer(model_one, layer_name)$output)
# beta_0 = -0.03$, and beta_1 = beta_2 = 0.9683
intermediate_layer_model$weights

# With 100
model_one %>% 
    fit(
      x = x, y = y,
      epochs = 100,
      validation_split = 0,
      verbose = 2,
      shuffle = FALSE
    )

layer_name <- 'Linear'
intermediate_layer_model <- keras_model(inputs = model_one$input,
                                        outputs = get_layer(model_one, layer_name)$output)
intermediate_layer_model$weights