"""
Jujutsu (jj) VCS Operation RL Environment

Usage:
    docker build -t jj-exec-server environments/community/jj_env/
    docker run -p 5003:5003 jj-exec-server
    python -m environments.community.jj_env.jj_env serve
"""


def __getattr__(name):
    if name == "JJEnv":
        from .jj_env import JJEnv

        return JJEnv
    if name == "JJEnvConfig":
        from .jj_env import JJEnvConfig

        return JJEnvConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["JJEnv", "JJEnvConfig"]
