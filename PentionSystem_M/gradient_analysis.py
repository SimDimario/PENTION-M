def estimate_gradient(observations):

    gradients = []

    for i in range(1, len(observations)):

        prev = observations[i - 1]

        curr = observations[i]

        delta = (
            curr.total_load -
            prev.total_load
        )

        gradients.append({

            "from":
                prev.sample_id,

            "to":
                curr.sample_id,

            "delta":
                delta,

            "trend":
                "increasing"
                if delta > 0
                else "decreasing",
        })

    return gradients