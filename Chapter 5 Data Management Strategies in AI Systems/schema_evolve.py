"""
Schema Evolution Demo
Shows how to handle data schema changes with versioning and migration
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
import json

class SchemaVersion(Enum):
    """Track schema versions"""
    V1 = "1.0"
    V2 = "2.0"
    V3 = "3.0"

@dataclass
class UserSchema_v1:
    """Initial user schema"""
    name: str
    email: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": SchemaVersion.V1.value,
            "data": asdict(self)
        }

@dataclass
class UserSchema_v2:
    """Added age and username fields"""
    name: str
    email: str
    age: Optional[int] = None
    username: Optional[str] = None

    @classmethod
    def migrate_from_v1(cls, v1: UserSchema_v1) -> 'UserSchema_v2':
        """Migrate data from v1 to v2"""
        return cls(
            name=v1.name,
            email=v1.email,
            age=None,
            username=v1.email.split('@')[0]  # Generate username from email
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": SchemaVersion.V2.value,
            "data": asdict(self)
        }

@dataclass
class UserSchema_v3:
    """Added metadata and updated fields"""
    name: str
    email: str
    username: str
    age: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool = True

    @classmethod
    def migrate_from_v2(cls, v2: UserSchema_v2) -> 'UserSchema_v3':
        """Migrate data from v2 to v3"""
        now = datetime.now()
        return cls(
            name=v2.name,
            email=v2.email,
            username=v2.username or v2.email.split('@')[0],
            age=v2.age,
            created_at=now,
            updated_at=now,
            is_active=True
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert datetime objects to ISO format
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        return {
            "version": SchemaVersion.V3.value,
            "data": data
        }

class UserSchemaManager:
    """Manages schema versions and migrations"""
    
    @staticmethod
    def migrate_to_latest(data: Dict[str, Any]) -> UserSchema_v3:
        """Migrate any version to the latest schema"""
        version = data.get("version", SchemaVersion.V1.value)
        
        if version == SchemaVersion.V1.value:
            v1 = UserSchema_v1(**data["data"])
            v2 = UserSchema_v2.migrate_from_v1(v1)
            return UserSchema_v3.migrate_from_v2(v2)
            
        elif version == SchemaVersion.V2.value:
            v2 = UserSchema_v2(**data["data"])
            return UserSchema_v3.migrate_from_v2(v2)
            
        elif version == SchemaVersion.V3.value:
            return UserSchema_v3(**data["data"])
            
        else:
            raise ValueError(f"Unknown schema version: {version}")

def demonstrate_schema_evolution():
    """Demonstrate schema evolution and migration"""
    
    print("\nSchema Evolution Demo")
    print("=" * 50)
    
    # Create a v1 user
    user_v1 = UserSchema_v1(
        name="John Doe",
        email="john@example.com"
    )
    print("\nV1 User:")
    print(json.dumps(user_v1.to_dict(), indent=2))
    
    # Migrate to v2
    user_v2 = UserSchema_v2.migrate_from_v1(user_v1)
    print("\nMigrated to V2:")
    print(json.dumps(user_v2.to_dict(), indent=2))
    
    # Migrate to v3
    user_v3 = UserSchema_v3.migrate_from_v2(user_v2)
    print("\nMigrated to V3:")
    print(json.dumps(user_v3.to_dict(), indent=2))
    
    # Demonstrate automatic migration to latest
    print("\nAutomatic Migration to Latest:")
    old_data = {
        "version": "1.0",
        "data": {
            "name": "Jane Smith",
            "email": "jane@example.com"
        }
    }
    
    latest_user = UserSchemaManager.migrate_to_latest(old_data)
    print(json.dumps(latest_user.to_dict(), indent=2))

if __name__ == "__main__":
    demonstrate_schema_evolution()
