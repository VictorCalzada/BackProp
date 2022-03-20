lin <- read.csv("lineal.csv",header = F)
ind=which(lin[,3]==0)
x_lin<-as.matrix(lin[,-3])
y_lin<-as.matrix(lin[,3])

cir <- read.csv("circle.csv",header = F)
ind=which(cir[,3]==0)
x_cir<-as.matrix(cir[,-3])
y_cir<-as.matrix(cir[,3])

activation <- function(z) {
    return(1/(1 + exp(-z)))
}

mse <- function(a,b){
    error<- a-b
    return(sum(error^2)/nrow(error))
}

backprop_mlp_one <- function(y, x, h, epochs = 100, eta = 0.1){
    x = cbind(x, rep(1,nrow(x))) #Poniendo el bias
    
    neurons <- c(ncol(x),h,ncol(y))
    n = length(neurons)-1
    W = list()
    for (i in 1:n){
        W[[i]] = matrix(data = runif((neurons[i])*neurons[i+1], min = -1, max = 1),
                 nrow = neurons[i+1], ncol = neurons[i])
    }
    for (i in 1:epochs){
        out = list()
        out[[1]] = activation(x %*% t(W[[1]]))
        for (o in 2:length(W)){
            out[[o]] = activation(out[[o-1]] %*% t(W[[o]]))
        }
        error = y - out[[2]]
        d = error*(1-out[[2]])*out[[2]]
        dw2 = t(d) %*% out[[1]]
        dw1 = t((d %*% W[[2]]) * ((1-out[[1]])*out[[1]])) %*% x
        W[[1]] = W[[1]] + eta * dw1
        W[[2]] = W[[2]] + eta * dw2 
    }
    return(out)
    
}

out = backprop_mlp_one(y_lin,x_lin,h=3,epochs = 1000)

cbind(out[[2]],y_lin)

eta <- c(0.001,0.01,0.1,0.5,0.75,1,2,5,10)
error <- c()
err = 1000
capas=0
etas=0
for (i in 1:20){
    for (o in eta){
        out = backprop_mlp_one(y_cir,x_cir,h=i,epochs = 1000,eta = o)
        error <- c(error, mse(out[[2]],y_cir))
        errNow <- mse(out[[2]],y_cir)
        if (errNow < err){
            err = errNow
            capas = i
            etas = o
        }
    }
}
c(capas,etas,err)

backprop_mlp_one_inercia <- function(y, x, h, epochs = 100, eta = 0.1, inercia = 0.1){
    x = cbind(x, rep(1,nrow(x))) #Poniendo el bias
    
    neurons <- c(ncol(x),h,ncol(y))
    n = length(neurons)-1
    W = list()
    for (i in 1:n){
        W[[i]] = matrix(data = runif((neurons[i])*neurons[i+1], min = -1, max = 1),
                 nrow = neurons[i+1], ncol = neurons[i])
    }
    for (i in 1:epochs){
        out = list()
        out[[1]] = activation(x %*% t(W[[1]]))
        for (o in 2:length(W)){
            out[[o]] = activation(out[[o-1]] %*% t(W[[o]]))
        }
        error = y - out[[2]]
        d = error*(1-out[[2]])*out[[2]]
        if (i == 1){
            liner = list()
            liner[[1]] = 0
            liner[[2]] = 0
        }
        
        dw2 = t(d) %*% out[[1]] + inercia +liner[[2]]
        dw1 = t((d %*% W[[2]]) * ((1-out[[1]])*out[[1]])) %*% x +inercia+liner[[1]]
        
        liner[[1]] = dw1
        liner[[2]] = dw2
        
        W[[1]] = W[[1]] + eta * dw1
        W[[2]] = W[[2]] + eta * dw2 
    }
    return(out)
    
}

eta <- c(0.001,0.01,0.1,0.5,0.75,1,2,5,10)
error <- c()
err = 1000
capas=0
etas=0
for (i in 1:20){
    for (o in eta){
        out = backprop_mlp_one(y_cir,x_cir,h=i,epochs = 1000,eta = o)
        error <- c(error, mse(out[[2]],y_cir))
        errNow <- mse(out[[2]],y_cir)
        if (errNow < err){
            err = errNow
            capas = i
            etas = o
        }
    }
}
c(capas,etas,err)

activation <- function(z) {
    1/(1 + exp(-z))
}



comOut <- function(x, W){
    #Funcion que computa el output de la red
    nW = length(W)
    val = cbind(x,rep(1,nrow(x))) #Valor inicial a multiplicar por W y activar 
    A = list()
    for (i in 1:nW){
        
        
        val = activation(val %*% t(W[[i]]))
        if (i != nW){
            val = cbind(val,rep(1,nrow(val)))
        }
        A[[i]] = val
        
        
    }
    
    return(A)
    
}

backprop_mlp <- function(y, x, h, epochs = 10, eta = 0.1){
    neurons = c(ncol(x),h,ncol(y))
    
    set.seed(888)
    n = length(neurons)-1
    W = list()
    for (i in 1:n){
        W[[i]] = matrix(data = runif((neurons[i]+1)*neurons[i+1], min = -1, max = 1),
                 nrow = neurons[i+1], ncol = neurons[i]+1)
    }
    
    
    
    # Dentro del for
    d = 0
    
    
    ex = length(neurons)
    
    for (i in 1:epochs){
        Wdelta = list()
        A = comOut(x,W)
        for (i in 1:n){
            o = ex - i
            if (o == n){
                d = ((y-A[[o]])*A[[o]]*(1-A[[o]]))
                
            }else{
                Wf = W[[o+1]]
                len = dim(Wf)[2]
                Af = A[[o]]
                
                if (dim(Wf)[1]==1){
                    Wf = Wf[-len]
                }else{
                    Wf = Wf[,-len]
                }
                
                d = d%*%Wf*Af[,-len]*(1-Af[,-len])
            }
            if (o == 1){
                Wdelta[[i]] = t(d)%*%cbind(x,rep(1,nrow(x)))#W[[o]] = W[[o]]+eta * t(d)%*%cbind(x,rep(1,nrow(x)))
            }else{
                Wdelta[[i]] = t(d)%*%A[[o-1]]#W[[o]] = W[[o]]+eta * t(d)%*%A[[o-1]]
            }
        }
        W <- mapply(function(X,Y){
        return(X + eta * Y)
        }, W, rev(Wdelta))
    }
    
    ###Fin del for
    
    
    return(A)
    
    
}

A = backprop_mlp(y_cir,x_cir,c(15,10,15),epochs = 1000, eta = 0.1)
n = length(A)
val <- cbind(A[[n]],y_cir)

val
