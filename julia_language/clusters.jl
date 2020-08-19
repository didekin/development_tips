# Previousl, in REPL:  ] add https://github.com/VMLS-book/VMLS.jl
using Pkg
Pkg.add(["LinearAlgebra", "Clustering", "Plots"])
using LinearAlgebra, VMLS, Plots, Clustering

# Clustering objective function
Jclust(x,reps,assignment) = avg([norm(x[i]-reps[assignment[i]])^2 for i=1:length(x)])
# Example:
x = [[0,1], [1,0], [-1,1]]
reps = [[1,1],[0,0]]
assignment = [1,2,1]
Jclust(x, reps, assignment)
assignment = [1,1,2]
Jclust(x,reps,assignment)

# K-means
function kmeans_vmls(x, k; maxiters = 100, tol = 1e-5)
    # N vectors in array x
    N = length(x)
    # n elements in each vector
    n = length(x[1])
    # To store the distance of each vector to the nearest representative
    distances = zeros(N)
    # k elements array of n-zeros vectors to store the k representatives
    reps = [zeros(n) for j = 1:k]
    # ’assignment’ is a vector of N integers between 1 and k, initially chosen randomly
    assignment = [rand(1:k) for i in 1:N]
    # Initial stopping condition. Inf denotes positive infinity.
    Jprevious = Inf
    for iter in 1:maxiters
        # Cluster j representative is average of points in cluster j.
        for j = 1:k
            # Indexes of vectors in array x s.t. their assignment == iteration index.
            group = [i for i=1:N if assignment[i] == j]
            # Mean of the vector in a group; itself a vector of length n.
            reps[j] = sum(x[group]) / length(group);
        end;
        for i = 1:N
            # findmin return the minimun distance of a vector to a representative and the index of the representative.
            # With those two values, distances and assingments vectors are updated.
            (distances[i], assignment[i]) = findmin([norm(x[i] - reps[j]) for j = 1:k])
        end;
        # Compute clustering objective: average of squared distances.
        J = norm(distances)^2 / N
        # Show progress
        println("Iteration ", iter, ": Jclust = ", J, ".")
        # Terminate if J stopped decreasing.
        if (iter > 1) && (abs(J-Jprevious) < tol*J)
            return assignment, reps
        end
        Jprevious = J
    end
end
# EXAMPLE
# An array of 300 2-elements vectors.
N = 300
k = 3
X = vcat([0.3*randn(2) for i = 1:(N/3)], [[1,1] + 0.3*randn(2) for i = 1:(N/3)],[[1,-1] + 0.3*randn(2) for i = 1:(N/3)])
scatter([x[1] for x in X], [x[2] for x in X])
plot!(size = (500,500))
plot!(legend = false, grid = false)
plot!(xlims = (-1.5,2.5), ylims = (-2,2))
# To call the function:
assignment, representatives = kmeans_vmls(X, 3)
# We extract 3 arrays with the vectors in each of the 3 assignment categories:
grps = [[X[i] for i=1:N if assignment[i] == j] for j=1:k]
# Occurrences in each group:
occurr = [length(grps[i]) for i in 1:k]
# Scatter plot of cluster points.
scatter([x_gr[1] for x_gr in grps[1]], [x_gr[2] for x_gr in grps[1]])
scatter!([x_gr[1] for x_gr in grps[2]], [x_gr[2] for x_gr in grps[2]])
scatter!([x_gr[1] for x_gr in grps[3]], [x_gr[2] for x_gr in grps[3]])
plot!(legend = false, grid = false, size = (500,500), xlims = (-1.5,2.5), ylims = (-2,2))
