class TrajectoryObservation:

    def __init__(

        self,

        sample_id,

        lat,

        lon,

        timestamp,

        concentrations,
    ):

        self.sample_id = sample_id

        self.lat = lat

        self.lon = lon

        self.timestamp = timestamp

        self.concentrations = {

            k.upper(): float(v)

            for k, v in concentrations.items()
        }

        self.total_load = sum(
            self.concentrations.values()
        )

        self.dominant_compound = max(
            self.concentrations,
            key=self.concentrations.get
        )


def reconcile_trajectory(request):

    observations = []

    for point in request.sampling_points:

        obs = TrajectoryObservation(

            sample_id=point.id,

            lat=point.coordinates.latitude,

            lon=point.coordinates.longitude,

            timestamp=point.sample_time,

            concentrations=
                point.measured_concentrations,
        )

        observations.append(obs)

    observations.sort(
        key=lambda x: x.timestamp
    )

    return observations