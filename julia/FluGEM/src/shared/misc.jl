using Dates

"""
closest_utime(obsv_day, u_times)

Return the element of u_times closest to obsv_day.
"""
function closest_utime(obsv_day::Int, u_times::Vector{Int})
    best_time = u_times[1]
    best_diff = abs(best_time - obsv_day)

    for t in @view u_times[2:end]
        d = abs(t - obsv_day)
        if d < best_diff
            best_diff = d
            best_time = t
        end
    end

    return best_time
end
