# SVM #

# using Pkg
using LinearAlgebra
using Random
# using StatsBase
Random.seed!(520)

# read data
data = open("C:/Users/dpelt/OneDrive - 서울시립대학교/Desktop/Mayson/UOS_graduate/julia/diabetes.txt")
# data = open("C:/Users/dpelt/OneDrive - 서울시립대학교/Desktop/Mayson/UOS_graduate/julia/diabetes_scaled.txt")
lines = readlines(data)

# data setting
n = length(lines)
p = length(split(lines[1]))
data = zeros(n, p)
for i in 1:length(lines)
    line = split(lines[i])
    data[i, p] = parse(Float64, line[1])
    for j in 2:p
        data[i, (j-1)] = parse(Float64, line[j][3:end])
    end
end
X = data[:, 1:p-1]
y = data[:, end]
p -= 1

# data scaling (?)
# for i in 1:p
#     X[:, i] = normalize!(X[:, i])
# end
# norm(X[:, 1])

# parameters
C = 1000
tol = 1e-2
eps = 1e-3

##########################################################
# initalize
alpha = zeros(n, 1)
b = 0 # threshold

non_bound_alpha = fill(false, n)
error_cache = zeros(n, 1)

#######################################################
# objective function
function objective(alpha)
    result = sum(alpha)
    for i in 1:n
        result += -sum(y[i] .* y .* (X * X[i, :]) .* alpha[i] .* alpha) ./ 2
    end
    return result
end

####################################################
# output function
function output_function(i)
    return sum(alpha .* y .* (X * X[i, :])) - b
end

#######################################################
# takestep
function takestep(i1, i2, a2, y2, E2)
    if (i1 == i2) 
        return false
    end
    a1 = alpha[i1]
    y1 = y[i1]
    E1 = output_function(i1) - y1
    s = y1 * y2
    if (s != 1)
        L = max(0, a2 - a1)
        H = min(C, C + a2 - a1)
    elseif (s == 1)
        L = max(0, a1 + a2 - C)
        H = min(C, a1 + a2)
    end 

    if (L == H)
        return false
    end

    k11 = dot(X[i1, :], X[i1, :])
    k12 = dot(X[i1, :], X[i2, :])
    k22 = dot(X[i2, :], X[i2, :])
    eta = 2*k12 - k11 - k22
    if (eta < 0)
        new_a2 = a2 - y2 * (E1 - E2) / eta
        if (new_a2 < L) 
            new_a2 = L
        elseif (new_a2 > H)
            new_a2 = H
        end
    else
        temp_alpha = deepcopy(alpha)
        temp_alpha[i2] = L
        L_obj = objective(temp_alpha)
        temp_alpha = deepcopy(alpha)
        temp_alpha[i2] = H
        H_obj = objective(temp_alpha)

        if (L_obj > (H_obj + eps))
            new_a2 = L
        elseif (L_obj < (H_obj - eps))
            new_a2 = H
        else 
            new_a2 = a2
        end
    end

    if (new_a2 < 1e-3)
        new_a2 = 0
    elseif (new_a2 > C - 1e-3)
        new_a2 = C
    end

    if (abs(new_a2 - a2) < eps * (new_a2 + a2 + eps))
        return false
    end

    new_a1 = a1 + s * (a2 - new_a2)

    if (((0 < new_a1) & (new_a1 < C)) & !(0 < new_a2) & (new_a2 < C))
        new_b = E1 + y1 * (new_a1 - a1) * dot(X[i1, :], X[i1, :]) + y2 * (new_a2 - a2) * dot(X[i1, :], X[i2, :]) + b
        non_bound_alpha[i2] = false
    elseif (!((0 < new_a1) & (new_a1 < C)) & (0 < new_a2) & (new_a2 < C))
        new_b = E2 + y1 * (new_a1 - a1) * dot(X[i1, :], X[i2, :]) + y2 * (new_a2 - a2) * dot(X[i2, :], X[i2, :]) + b
        non_bound_alpha[i1] = false
    else
        b1 = E1 + y1 * (new_a1 - a1) * dot(X[i1, :], X[i1, :]) + y2 * (new_a2 - a2) * dot(X[i1, :], X[i2, :]) + b
        b2 = E2 + y1 * (new_a1 - a1) * dot(X[i1, :], X[i2, :]) + y2 * (new_a2 - a2) * dot(X[i2, :], X[i2, :]) + b
        new_b = (b1 + b2) / 2
    end

    # for j in findall(!iszero, non_bound_alpha)
    #     if ((j != i1) || (j != i2))
    #         error_cache[j] = error_cache[j] + y1 * (new_a1 - a1) * dot(X[i1, :], X[j, :]) + 
    #             y2 * (new_a2 - a2) * dot(X[i2, :], X[j, :]) + b - new_b
    #     end
    # end

    # println(new_a1, new_a2)
    global b = new_b
    global alpha[i1] = new_a1
    global alpha[i2] = new_a2
    return true
end

####################################################
# second choice heuristic
function second_choice(E2)
    max_step_size = 0
    max_idx = 0
    for i in 1:n
        E1 = output_function(i)
        if(max_step_size < abs(E2 - E1))
            max_step_size = abs(E2 - E1)
            max_idx = i
        end
    end
    return max_idx
end

####################################################
# examine_example
function examine_example(i2)
    y2 = y[i2]
    a2 = alpha[i2]
    E2 = output_function(i2) - y2
    r2 = E2 * y2 
    if (((r2 < -tol) & (a2 < C)) | ((r2 > tol) & (a2 > 0)))
        if (sum(non_bound_alpha) > 1)
            # second choice heuristic part must be added ?
            i1 = second_choice(E2)
            if takestep(i1, i2, y2, a2, E2)
                return 1
            end
        end
        for i in shuffle(findall(!iszero, non_bound_alpha))
            i1 = deepcopy(i)
            if takestep(i1, i2, y2, a2, E2)
                return 1
            end
        end
        for i in shuffle(collect(1:n))
            i1 = deepcopy(i)
            if takestep(i1, i2, y2, a2, E2)
                return 1
            end
        end
    end
    return 0
end


#############################################3
# main
num_changed = 0
examine_all = true
while((num_changed > 0) | examine_all)
    global num_changed = 0
    if (examine_all)
        for i in 1:n
            global num_changed = num_changed + examine_example(i)
        end
    else
        for i in findall(!iszero, non_bound_alpha)
            global num_changed = num_changed + examine_example(i)
        end
    end

    if (examine_all)
        global examine_all = false
    elseif (num_changed == 0)
        global examine_all = true
    end
    println(sum(alpha))
end