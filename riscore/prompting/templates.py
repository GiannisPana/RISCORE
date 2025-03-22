"""Prompt templates for RISCORE framework."""

from pathlib import Path
from typing import Dict, Optional
import yaml
import json


class PromptTemplate:
    """Single prompt template."""
    
    def __init__(self, name: str, system: str, user: str, metadata: Optional[Dict] = None):
        self.name = name
        self.system = system
        self.user = user
        self.metadata = metadata or {}
    
    def format(self, **kwargs) -> tuple[str, str]:
        """Format template with variables."""
        return (
            self.system.format(**kwargs),
            self.user.format(**kwargs)
        )


class PromptTemplateManager:
    """Manage and load prompt templates."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize prompt template manager.
        
        Args:
            templates_dir: Directory containing template files
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, PromptTemplate] = {}
        
        if self.templates_dir.exists():
            self._load_templates()
    
    def _load_templates(self):
        """Load all templates from directory."""
        for file_path in self.templates_dir.glob("*.yaml"):
            self._load_template_file(file_path)
        
        for file_path in self.templates_dir.glob("*.json"):
            self._load_template_file(file_path)
    
    def _load_template_file(self, file_path: Path):
        """Load templates from a single file."""
        if file_path.suffix == ".yaml" or file_path.suffix == ".yml":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        elif file_path.suffix == ".json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            return
        
        # Support both single template and multiple templates in one file
        if isinstance(data, dict) and "templates" in data:
            templates_data = data["templates"]
        elif isinstance(data, list):
            templates_data = data
        else:
            templates_data = [data]
        
        for template_data in templates_data:
            template = PromptTemplate(
                name=template_data["name"],
                system=template_data["system"],
                user=template_data["user"],
                metadata=template_data.get("metadata", {})
            )
            self.templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        return self.templates.get(name)
    
    def list_templates(self) -> list[str]:
        """List all available template names."""
        return list(self.templates.keys())
    
    def add_template(self, template: PromptTemplate):
        """Add a template programmatically."""
        self.templates[template.name] = template
    
    def save_template(self, template: PromptTemplate, filename: Optional[str] = None):
        """Save template to file."""
        self.templates_dir.mkdir(exist_ok=True, parents=True)
        
        if filename is None:
            filename = f"{template.name}.yaml"
        
        filepath = self.templates_dir / filename
        
        data = {
            "name": template.name,
            "system": template.system,
            "user": template.user,
            "metadata": template.metadata,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# Global template manager instance
_template_manager: Optional[PromptTemplateManager] = None


def get_template_manager() -> PromptTemplateManager:
    """Get global template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    return _template_manager


def get_template(name: str) -> Optional[PromptTemplate]:
    """Get template by name from global manager."""
    return get_template_manager().get_template(name)
