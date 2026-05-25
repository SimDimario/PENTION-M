def estimate_hotspot(observations):

    hotspot = max(

        observations,

        key=lambda x: x.total_load
    )

    return {

        "latitude":
            hotspot.lat,

        "longitude":
            hotspot.lon,

        "dominant_compound":
            hotspot.dominant_compound,

        "total_load":
            hotspot.total_load,

        "sample_id":
            hotspot.sample_id,
    }