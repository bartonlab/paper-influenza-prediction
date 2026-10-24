# ---------------------------------------------------------------------------
# BioTools.hamming
# ---------------------------------------------------------------------------
# Local copy of the only function this project used from the unregistered
# FluPredictibility package (`FluPredictibility.BioTools.hamming`), with the same
# signature and behaviour, so the project installs from the General registry alone.
# Callers keep writing `BioTools.hamming(a, b)`.

module BioTools

using BioSequences: BioSequence, isambiguous

export hamming

function hamming_ignore_ambiguous(a::BioSequence, b::BioSequence)
    h = 0
    for (x, y) in zip(a, b)
        if x != y && !isambiguous(x) && !isambiguous(y)
            h += 1
        end
    end
    return h
end

function hamming_ambiguous_mismatch(a::BioSequence, b::BioSequence)
    h = 0
    for (x, y) in zip(a, b)
        if x != y
            h += 1
        end
    end
    return h
end

"""
    hamming(a::BioSequence, b::BioSequence, [normalization=false]; ambiguous=:ignore)

Number of positions where `a` and `b` differ, as a `Float64` (divided by the
sequence length when `normalization = true`). `ambiguous = :ignore` treats
ambiguous symbols as equal to anything; `:mismatch` treats them as regular symbols.
"""
function hamming(a::BioSequence, b::BioSequence, normalization = false; ambiguous = :ignore)
    if ambiguous == :ignore && normalization
        return hamming_ignore_ambiguous(a, b) / length(a)
    elseif ambiguous == :ignore && !normalization
        return Float64(hamming_ignore_ambiguous(a, b))
    elseif ambiguous == :mismatch && normalization
        return hamming_ambiguous_mismatch(a, b) / length(a)
    elseif ambiguous == :mismatch && !normalization
        return Float64(hamming_ambiguous_mismatch(a, b))
    else
        @error "Incorrect arguments."
    end
end

end # module BioTools
