import os
import requests
import streamlit as st

BACKEND_URL = st.secrets.get(
    "BACKEND_URL",
    os.getenv("BACKEND_URL", "https://financify-saas.onrender.com")
).rstrip("/")

SURECART_CHECKOUT_URL = st.secrets.get(
    "SURECART_CHECKOUT_URL",
    os.getenv("SURECART_CHECKOUT_URL", "https://financify.blog/buy/financify-tools")
)

DEV_MODE = str(
    st.secrets.get("DEV_MODE", os.getenv("DEV_MODE", "false"))
).lower() == "true"


def _post(path: str, payload: dict, timeout: int = 60):
    response = requests.post(
        f"{BACKEND_URL}{path}",
        json=payload,
        timeout=timeout
    )

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(str(detail))

    try:
        return response.json()
    except Exception:
        return {"ok": True, "raw": response.text}


def send_login_code(email: str):
    email = (email or "").strip().lower()
    if not email:
        raise RuntimeError("Please enter your email.")
    return _post("/auth/send-code", {"email": email})


def verify_login_code(email: str, code: str):
    email = (email or "").strip().lower()
    code = (code or "").strip()

    if not email:
        raise RuntimeError("Please enter your email.")
    if not code:
        raise RuntimeError("Please enter the login code.")

    return _post("/auth/verify-code", {"email": email, "code": code})


def check_subscription(email: str):
    email = (email or "").strip().lower()

    if DEV_MODE:
        return {
            "active": True,
            "plan": "DEV MODE",
            "email": email or "dev@financify.blog"
        }

    if not email:
        return {"active": False, "plan": "Free"}

    try:
        return _post("/subscription/check", {"email": email})
    except Exception:
        return {"active": False, "plan": "Free"}


def logout():
    for key in ["user_email", "logged_in", "subscription", "login_code_sent"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def protect_app():
    if DEV_MODE:
        st.session_state["logged_in"] = True
        st.session_state["user_email"] = "dev@financify.blog"
        st.session_state["subscription"] = {
            "active": True,
            "plan": "DEV MODE"
        }
        return True

    if st.session_state.get("logged_in") and st.session_state.get("user_email"):
        sub = check_subscription(st.session_state["user_email"])
        st.session_state["subscription"] = sub

        if sub.get("active"):
            return True

        st.warning("Your subscription is not active.")
        st.markdown(
            f"""
            <a href="{SURECART_CHECKOUT_URL}" target="_blank">
                <button style="
                    background:#f5b400;
                    color:#111;
                    border:none;
                    padding:0.85rem 1.2rem;
                    border-radius:14px;
                    font-weight:800;
                    cursor:pointer;">
                    Upgrade to Financify Tools
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )

        if st.button("Logout"):
            logout()

        st.stop()

    st.markdown("## 🔐 Login to Financify Tools")
    st.caption("Enter your email to receive a secure login code.")

    email = st.text_input("Email", placeholder="you@example.com")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Send Login Code", use_container_width=True):
            try:
                send_login_code(email)
                st.session_state["login_email"] = email.strip().lower()
                st.session_state["login_code_sent"] = True
                st.success("Login code sent. Please check your email.")
            except Exception as e:
                st.error(str(e))

    if st.session_state.get("login_code_sent"):
        code = st.text_input("Login Code", placeholder="Enter code")

        with col2:
            if st.button("Verify Code", use_container_width=True):
                try:
                    result = verify_login_code(
                        st.session_state.get("login_email", email),
                        code
                    )

                    user_email = result.get(
                        "email",
                        st.session_state.get("login_email", email)
                    )

                    st.session_state["logged_in"] = True
                    st.session_state["user_email"] = user_email.strip().lower()
                    st.session_state["subscription"] = check_subscription(user_email)

                    st.success("Login successful.")
                    st.rerun()

                except Exception as e:
                    st.error(str(e))

    st.stop()
