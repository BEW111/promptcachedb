import cProfile
import pstats
import functools


def cprofile_function_and_save(profile_filename="profile.prof"):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()

            # Call the function
            result = f(*args, **kwargs)

            pr.disable()
            pr.dump_stats(profile_filename)
            
            print(f"Profile data saved to {profile_filename}")
            return result

        return wrapper
    return decorator