def parse_blocks(name, s) -> set:
    vals = (rawval.strip() for rawval in s.split(","))
    return {(name, int(val.strip())) for val in vals if val}


def convert_time(ms, time_mode, start_time, end_time) -> tuple:
    match time_mode:
        case "sigma":
            return (start_time, end_time)
        case "percent" | "timestep":
            if time_mode == "timestep":
                start_time = 1.0 - (start_time / 999.0)
                end_time = 1.0 - (end_time / 999.0)
            else:
                if start_time > 1.0 or start_time < 0.0:
                    raise ValueError(
                        "invalid value for start percent",
                    )
                if end_time > 1.0 or end_time < 0.0:
                    raise ValueError(
                        "invalid value for end percent",
                    )
            return (ms.percent_to_sigma(start_time), ms.percent_to_sigma(end_time))
        case _:
            raise ValueError("invalid time mode")


def check_time(sigma, start_sigma, end_sigma):
    if sigma is None:
        return False
    sigma = sigma.detach().cpu().max().item()
    return sigma <= start_sigma and sigma >= end_sigma
