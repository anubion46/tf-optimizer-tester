library(plotly)

# Function 1: sum of powers
sum_powers <- function(x, y){
  abs(x) ** 2 + abs(y) ** 2
}

n <- 40
x <- seq(-1, 1, length.out = n)
y <- seq(-1, 1, length.out = n)
z <- matrix(nrow = n, ncol = n)
for (i in 1:n){
  for (j in 1:n)
    z[i,j] <- sum_powers(x[i], y[j])
}


# Function 2: sum of sin
sum_sin <- function(x, y){
  abs(sin(x) + sin(y))/2
}

n <- 100
x <- seq(-5, 5, length.out = n)
y <- seq(-5, 5, length.out = n)
z <- matrix(nrow = n, ncol = n)
for (i in 1:n){
  for (j in 1:n)
    z[i,j] <- 1 - sum_sin(x[i], y[j])
}


# Function 3: deb01
deb01 <- function(x, y){
  abs(sin(5.0 * pi * x) ** 6  + sin(5.0 * pi * y) ** 6)/2
}

n <- 100
x <- seq(-1, 1, length.out = n)
y <- seq(-1, 1, length.out = n)
z <- matrix(nrow = n, ncol = n)
for (i in 1:n){
  for (j in 1:n)
    z[i,j] <- deb01(x[i], y[j])
}


# Function 4: alpine01
alpine01 <- function(x, y){
  abs(x * sin(x) + 0.1 * x) + abs(y * sin(y) + 0.1 * y)
}

n <- 100
x <- seq(-10, 10, length.out = n)
y <- seq(-10, 10, length.out = n)
z <- matrix(nrow = n, ncol = n)
for (i in 1:n){
  for (j in 1:n)
    z[i,j] <- alpine01(x[i], y[j])
}


# Function 5: rastrigin
rastrigin <- function(x, y){
  20*((x**2 - 10*cos(2*pi*x) + 10) + (y**2 - 10*cos(2*pi*y) + 10))
}

n <- 200
x <- seq(-10, 10, length.out = n)
y <- seq(-10, 10, length.out = n)
z <- matrix(nrow = n, ncol = n)
for (i in 1:n){
  for (j in 1:n)
    z[i,j] <- rastrigin(x[i], y[j])
}


# Visualize
p <- plot_ly(x = x, y = y, z = z) %>% add_surface()
p %>% offline(height = 600)
