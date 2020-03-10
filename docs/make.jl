using Documenter, AttractorNetworksBase

makedocs(;
    modules=[AttractorNetworksBase],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/dylanfesta/AttractorNetworksBase.jl/blob/{commit}{path}#L{line}",
    sitename="AttractorNetworksBase.jl",
    authors="Dylan Festa",
    assets=String[],
)

deploydocs(;
    repo="github.com/dylanfesta/AttractorNetworksBase.jl",
)
