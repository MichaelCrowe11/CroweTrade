"""
Production-grade security and authentication for CroweTrade.
Includes JWT authentication, API key management, and security middleware.
"""

import os
import jwt
import bcrypt
import secrets
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import wraps
import hashlib
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

logger = logging.getLogger(__name__)


@dataclass
class User:
    """User account representation."""
    user_id: str
    username: str
    email: str
    roles: List[str]
    is_active: bool = True
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles
    
    def has_any_role(self, roles: List[str]) -> bool:
        """Check if user has any of the specified roles."""
        return any(role in self.roles for role in roles)


@dataclass
class APIKey:
    """API key representation."""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    permissions: List[str]
    is_active: bool = True
    created_at: datetime = None
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def is_expired(self) -> bool:
        """Check if the API key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if API key has a specific permission."""
        return permission in self.permissions


class PasswordManager:
    """Secure password hashing and validation."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def generate_secure_password(length: int = 16) -> str:
        """Generate a secure random password."""
        return secrets.token_urlsafe(length)


class EncryptionManager:
    """Handle data encryption and decryption."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        if encryption_key:
            self.key = encryption_key.encode()
        else:
            self.key = os.getenv('ENCRYPTION_KEY', '').encode()
            
        if len(self.key) == 0:
            raise ValueError("Encryption key is required")
            
        # Derive a proper Fernet key from the provided key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'salt_',  # In production, use a random salt stored securely
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.key))
        self.fernet = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise


class JWTManager:
    """JWT token management for authentication."""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
        
    def generate_token(self, user: User, expires_in: timedelta = None) -> str:
        """Generate a JWT token for a user."""
        if expires_in is None:
            expires_in = timedelta(hours=24)
            
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'roles': user.roles,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + expires_in,
            'jti': secrets.token_hex(16),  # JWT ID for token revocation
        }
        
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            return token
        except Exception as e:
            logger.error(f"Token generation error: {e}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh an existing token."""
        payload = self.verify_token(token)
        if not payload:
            return None
            
        # Create new token with extended expiry
        new_payload = payload.copy()
        new_payload['iat'] = datetime.utcnow()
        new_payload['exp'] = datetime.utcnow() + timedelta(hours=24)
        new_payload['jti'] = secrets.token_hex(16)
        
        try:
            new_token = jwt.encode(new_payload, self.secret_key, algorithm=self.algorithm)
            return new_token
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None


class APIKeyManager:
    """Manage API keys for authentication."""
    
    def __init__(self):
        self.api_keys: Dict[str, APIKey] = {}
    
    def create_api_key(self, user_id: str, name: str, permissions: List[str], 
                      expires_in: Optional[timedelta] = None) -> Tuple[str, APIKey]:
        """Create a new API key."""
        # Generate secure API key
        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Create expiry date if specified
        expires_at = None
        if expires_in:
            expires_at = datetime.now() + expires_in
            
        api_key = APIKey(
            key_id=secrets.token_hex(16),
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions,
            expires_at=expires_at
        )
        
        self.api_keys[key_hash] = api_key
        
        return raw_key, api_key
    
    def verify_api_key(self, raw_key: str) -> Optional[APIKey]:
        """Verify an API key."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        api_key = self.api_keys.get(key_hash)
        
        if not api_key:
            return None
            
        if not api_key.is_active:
            return None
            
        if api_key.is_expired():
            return None
            
        # Update last used timestamp
        api_key.last_used = datetime.now()
        return api_key
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        for api_key in self.api_keys.values():
            if api_key.key_id == key_id:
                api_key.is_active = False
                return True
        return False


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self):
        self.requests: Dict[str, List[datetime]] = {}
        
    def is_allowed(self, client_id: str, limit: int, window_seconds: int = 3600) -> bool:
        """Check if a request is allowed under rate limits."""
        now = datetime.now()
        window_start = now - timedelta(seconds=window_seconds)
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id] 
                if req_time > window_start
            ]
        else:
            self.requests[client_id] = []
            
        # Check if under limit
        if len(self.requests[client_id]) >= limit:
            return False
            
        # Record this request
        self.requests[client_id].append(now)
        return True


class SecurityMiddleware:
    """Security middleware for API endpoints."""
    
    def __init__(self, jwt_manager: JWTManager, api_key_manager: APIKeyManager):
        self.jwt_manager = jwt_manager
        self.api_key_manager = api_key_manager
        self.rate_limiter = RateLimiter()
        
    def require_auth(self, required_roles: Optional[List[str]] = None):
        """Decorator to require authentication."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # This would integrate with your web framework
                # For now, it's a placeholder showing the logic
                auth_header = kwargs.get('auth_header', '')
                
                user = None
                
                # Try JWT authentication
                if auth_header.startswith('Bearer '):
                    token = auth_header[7:]
                    payload = self.jwt_manager.verify_token(token)
                    if payload:
                        user = User(
                            user_id=payload['user_id'],
                            username=payload['username'],
                            email='',  # Would be loaded from database
                            roles=payload['roles']
                        )
                
                # Try API key authentication
                elif auth_header.startswith('ApiKey '):
                    api_key = auth_header[7:]
                    key_obj = self.api_key_manager.verify_api_key(api_key)
                    if key_obj:
                        user = User(
                            user_id=key_obj.user_id,
                            username='',  # Would be loaded from database
                            email='',
                            roles=[]  # Would be loaded from database
                        )
                
                if not user:
                    raise PermissionError("Authentication required")
                    
                # Check role requirements
                if required_roles and not user.has_any_role(required_roles):
                    raise PermissionError("Insufficient permissions")
                    
                # Add user to kwargs
                kwargs['current_user'] = user
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_rate_limit(self, limit: int, window_seconds: int = 3600):
        """Decorator to enforce rate limiting."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                client_id = kwargs.get('client_ip', 'unknown')
                
                if not self.rate_limiter.is_allowed(client_id, limit, window_seconds):
                    raise PermissionError("Rate limit exceeded")
                    
                return func(*args, **kwargs)
            return wrapper
        return decorator


# Utility functions for request signing
def sign_request(method: str, url: str, body: str, secret: str) -> str:
    """Sign a request using HMAC."""
    message = f"{method.upper()}\n{url}\n{body}"
    signature = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return signature


def verify_request_signature(method: str, url: str, body: str, 
                           signature: str, secret: str) -> bool:
    """Verify a request signature."""
    expected_signature = sign_request(method, url, body, secret)
    return hmac.compare_digest(signature, expected_signature)


# Global instances - initialize with configuration
_jwt_manager = None
_api_key_manager = APIKeyManager()
_encryption_manager = None
_security_middleware = None


def initialize_security(jwt_secret: str, encryption_key: str):
    """Initialize security components."""
    global _jwt_manager, _encryption_manager, _security_middleware
    
    _jwt_manager = JWTManager(jwt_secret)
    _encryption_manager = EncryptionManager(encryption_key)
    _security_middleware = SecurityMiddleware(_jwt_manager, _api_key_manager)
    
    logger.info("Security components initialized")


def get_jwt_manager() -> JWTManager:
    """Get the JWT manager."""
    if _jwt_manager is None:
        raise RuntimeError("Security not initialized - call initialize_security() first")
    return _jwt_manager


def get_api_key_manager() -> APIKeyManager:
    """Get the API key manager."""
    return _api_key_manager


def get_encryption_manager() -> EncryptionManager:
    """Get the encryption manager."""
    if _encryption_manager is None:
        raise RuntimeError("Security not initialized - call initialize_security() first")
    return _encryption_manager


def get_security_middleware() -> SecurityMiddleware:
    """Get the security middleware."""
    if _security_middleware is None:
        raise RuntimeError("Security not initialized - call initialize_security() first")
    return _security_middleware