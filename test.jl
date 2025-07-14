using LinearAlgebra
using Statistics
using Random
using Printf
using Plots
using StatsBase

# Generates A and B matrices and their product C
function generate_A_B_and_C(n)
    A = zeros(n, n)
    B = zeros(n, n)

    for i in 1:n
        scale = rand()
        if scale < 0.5
            scale_A = scale * 0.1
            scale_B = scale * 0.1
        else
            scale_A = scale * 100.0
            scale_B = scale * 100.0
        end

        A[:, i] = randn(n) * scale_A
        B[i, :] = randn(n) * scale_B
    end

    C = A * B
    return A, B, C
end


# Computes the optimal probabilities
function optimal_probabilities(A, B)
    n = size(A, 2)
    ps = [norm(A[:, i]) * norm(B[i, :]) for i in 1:n]
    return ps ./ sum(ps)
end

# Estimates AB using s (number of samples) samples using probabilities
function estimate_product(A, B, probs, s)
    n = size(A, 2)
    C_est = zeros(size(A, 1), size(B, 2))
    for _ in 1:s
        i = sample(1:n, Weights(probs))
        C_est += (1 / (s * probs[i])) * (A[:, i] * B[i, :]')
    end
    return C_est
end

# Executes the experiment for diferent s (number of samples) values
function experiment(n, max_s, num_experiments)
    s_vals = 1:max_s
    optimal_errors_sum = zeros(Float64, max_s)
    unif_errors_sum = zeros(Float64, max_s)

    for exp in 1:num_experiments
        println("Experimento $exp/$num_experiments")
        A, B, C_real = generate_A_B_and_C(n)
        optimal_ps = optimal_probabilities(A, B)
        unif_ps = fill(1/n, n)

        for (idx, s) in enumerate(s_vals)
            C_approx_optimal = estimate_product(A, B, optimal_ps, s)
            C_approx_uniform = estimate_product(A, B, unif_ps, s)

            nrm = norm(C_real)
            err_optimal = norm(C_real - C_approx_optimal) / nrm
            err_uniform = norm(C_real - C_approx_uniform) / nrm

            optimal_errors_sum[idx] += err_optimal
            unif_errors_sum[idx] += err_uniform
        end
    end

    # Computes the mean
    optimal_errors_avg = optimal_errors_sum ./ num_experiments
    unif_errors_avg = unif_errors_sum ./ num_experiments

    return s_vals, optimal_errors_avg, unif_errors_avg
end


# Plots the results from the experiments
function plot_results(s_vals, optimal_errors, unif_errors)
    plot(s_vals, optimal_errors,
        label = "Amostragem Ótima",
        xlabel = "Número de Amostras (s)",
        ylabel = "Erro Relativo (escala log)",
        title = "Erro da Aproximação vs Número de Amostras",
        linewidth = 2,
        yscale = :log10)

    plot!(s_vals, unif_errors,
        label = "Amostragem Uniforme",
        linewidth = 2)
    savefig("result.png")
end

# Executes the experiment and the computes the variance for different number of samples values
function experiment_variance(n, max_s, num_experiments)
    s_vals = 1:max_s
    errors_optimal = [Float64[] for _ in s_vals]
    errors_uniform = [Float64[] for _ in s_vals]

    for exp in 1:num_experiments
        println("Experimento $exp/$num_experiments")
        A, B, C_real = generate_A_B_and_C(n)
        optimal_ps = optimal_probabilities(A, B)
        unif_ps = fill(1/n, n)

        for (idx, s) in enumerate(s_vals)
            C_approx_optimal = estimate_product(A, B, optimal_ps, s)
            C_approx_uniform = estimate_product(A, B, unif_ps, s)

            nrm = norm(C_real)
            err_optimal = norm(C_real - C_approx_optimal) / nrm
            err_uniform = norm(C_real - C_approx_uniform) / nrm

            push!(errors_optimal[idx], err_optimal)
            push!(errors_uniform[idx], err_uniform)
        end
    end

    var_optimal = [var(errors) for errors in errors_optimal]
    var_uniform = [var(errors) for errors in errors_uniform]

    return s_vals, var_optimal, var_uniform
end

# Plots the results from the variance experiment
function plot_variance(s_vals, var_optimal, var_uniform)
    plot(s_vals, var_optimal,
        label = "Amostragem Ótima",
        xlabel = "Número de Amostras (s)",
        ylabel = "Variância do Erro Relativo",
        title = "Variância do Erro vs Número de Amostras",
        linewidth = 2,
        yscale = :log10)

    plot!(s_vals, var_uniform,
        label = "Amostragem Uniforme",
        linewidth = 2)

    savefig("variance-result.png")
end



# main
n = 200          # matrix size
max_s = 200      # maximum number of samples
num_experiments = 100  # number of experiments
# s_vals, optimal_errors, unif_errors = experiment(n, max_s, num_experiments)
# plot_results(s_vals, optimal_errors, unif_errors)

s_vals, var_optimal, var_uniform = experiment_variance(n, max_s, num_experiments)
plot_variance(s_vals, var_optimal, var_uniform)

