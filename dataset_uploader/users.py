"""User accounts for the MLOps console.

Split out of app.py now that accounts are administrator-managed rather than
self-service: there are several operations on this table (create, list,
delete, promote/demote, password change), and keeping the SQL here means
app.py's routes stay about HTTP and permissions rather than queries.

Roles are deliberately a single boolean. This console has exactly two kinds
of user -- an administrator who manages accounts, and a member who can use
everything else -- and anything finer-grained would be invented complexity
for an internal tool with a handful of users.
"""
from sqlalchemy import text


def init_schema(engine):
    """Creates the table if absent, adds is_admin to deployments that predate
    admin accounts, and guarantees the console always has someone who can
    administer it.

    That last part matters for upgrades: an existing deployment already has
    accounts, all of which would default to is_admin = FALSE, locking
    everyone out of user management. So if no admin exists (a pre-existing
    deployment, or one whose only admin was removed directly in the
    database), the earliest-created account is promoted.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
        )
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE"))
        conn.execute(
            text(
                """
                UPDATE users SET is_admin = TRUE
                WHERE id = (SELECT MIN(id) FROM users)
                  AND NOT EXISTS (SELECT 1 FROM users WHERE is_admin)
                """
            )
        )


def get_by_username(engine, username):
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, username, password_hash, is_admin FROM users WHERE username = :u"),
            {"u": username},
        ).mappings().first()
    return dict(row) if row else None


def get_by_id(engine, user_id):
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, username, password_hash, is_admin FROM users WHERE id = :i"),
            {"i": user_id},
        ).mappings().first()
    return dict(row) if row else None


def list_all(engine):
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, username, is_admin, created_at FROM users ORDER BY id")
        ).mappings().all()
    return [dict(row) for row in rows]


def count(engine):
    with engine.begin() as conn:
        return conn.execute(text("SELECT count(*) FROM users")).scalar_one()


def admin_count(engine):
    with engine.begin() as conn:
        return conn.execute(text("SELECT count(*) FROM users WHERE is_admin")).scalar_one()


def create(engine, username, password_hash, is_admin=False):
    """Raises if the username is taken (the UNIQUE constraint) -- callers
    render that as a form error rather than checking first, so two
    simultaneous creates can't both pass a pre-check and then collide.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO users (username, password_hash, is_admin) "
                "VALUES (:u, :p, :a)"
            ),
            {"u": username, "p": password_hash, "a": is_admin},
        )


def delete(engine, user_id):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM users WHERE id = :i"), {"i": user_id})


def set_admin(engine, user_id, is_admin):
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE users SET is_admin = :a WHERE id = :i"),
            {"a": is_admin, "i": user_id},
        )


def set_password(engine, user_id, password_hash):
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE users SET password_hash = :p WHERE id = :i"),
            {"p": password_hash, "i": user_id},
        )
