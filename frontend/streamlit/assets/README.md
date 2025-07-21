# RAG LlamaStack - Bootstrap UI Components

## Overview

This directory contains the simplified Bootstrap-based UI components for the RAG LlamaStack application. We've moved from a complex multi-file CSS setup to a single, clean Bootstrap approach.

## Files

### `bootstrap-app.css`
- **Single source of styling** using Bootstrap 5.3.0
- Minimal custom CSS overrides
- Responsive design with Bootstrap grid system
- Clean, modern UI components

### `bootstrap_components.py`
- **Reusable Bootstrap components** for Streamlit
- Consistent styling across the application
- Easy to maintain and extend
- Uses Bootstrap classes for all styling

## Key Benefits

### ✅ Simplified Architecture
- **Single CSS file** instead of multiple scattered files
- **Bootstrap framework** provides consistent, tested components
- **Minimal custom CSS** - mostly Bootstrap class usage

### ✅ Better Maintainability
- **Standard Bootstrap classes** - no custom CSS to debug
- **Component-based approach** - reusable UI elements
- **Clear separation** between logic and styling

### ✅ Improved Performance
- **CDN-loaded Bootstrap** - faster loading
- **Optimized components** - less custom CSS to parse
- **Responsive by default** - Bootstrap handles mobile

### ✅ Modern UI/UX
- **Professional appearance** with Bootstrap design system
- **Consistent spacing** and typography
- **Accessible components** following Bootstrap standards

## Usage

### In Streamlit Components
```python
from components.bootstrap_components import render_alert, render_metrics_grid

# Use Bootstrap alerts
render_alert("Success message", "success")
render_alert("Warning message", "warning")

# Use Bootstrap metrics
metrics = [
    {"label": "Documents", "value": "5"},
    {"label": "Size", "value": "2.3 MB"}
]
render_metrics_grid(metrics)
```

### CSS Classes Available
- **Bootstrap 5.3.0** - All standard Bootstrap classes
- **Custom overrides** - Minimal Streamlit-specific styling
- **Responsive utilities** - Mobile-first design

## Migration from Old System

### What Changed
- ❌ Removed: `styles.css`, `template-components.css`, `tailwind.css`
- ✅ Added: `bootstrap-app.css`, `bootstrap_components.py`
- 🔄 Updated: All components to use Bootstrap classes

### Benefits Achieved
- **90% less custom CSS** - mostly Bootstrap classes now
- **Faster development** - standard components available
- **Better consistency** - Bootstrap design system
- **Easier maintenance** - single source of truth

## Bootstrap Components Used

### Alerts & Notifications
- `alert alert-success` - Success messages
- `alert alert-warning` - Warning messages  
- `alert alert-danger` - Error messages
- `alert alert-info` - Info messages

### Cards & Containers
- `card` - Document cards, model status
- `card-header` - Card headers with background
- `card-body` - Card content areas

### Progress & Status
- `progress` - File upload progress
- `badge` - Status indicators
- `spinner-border` - Loading indicators

### Layout & Grid
- `row` - Bootstrap grid system
- `col-md-3` - Responsive columns
- `container-fluid` - Full-width containers

### Forms & Inputs
- `form-control` - Input styling
- `btn btn-primary` - Button styling
- `input-group` - Input groups

## Customization

### Adding New Components
1. Create function in `bootstrap_components.py`
2. Use Bootstrap classes for styling
3. Keep custom CSS minimal
4. Test responsive behavior

### Theme Customization
- Modify CSS variables in `bootstrap-app.css`
- Use Bootstrap's theming system
- Override only when necessary

## Best Practices

### ✅ Do
- Use Bootstrap classes first
- Keep custom CSS minimal
- Test responsive behavior
- Follow Bootstrap conventions

### ❌ Don't
- Write custom CSS when Bootstrap has a solution
- Override Bootstrap styles unnecessarily
- Mix different CSS frameworks
- Ignore responsive design

## Future Enhancements

### Planned Improvements
- **Dark mode support** with Bootstrap 5.3.0
- **More interactive components** using Bootstrap JS
- **Advanced animations** with Bootstrap utilities
- **Accessibility improvements** following Bootstrap guidelines

### Component Library
- **Reusable UI patterns** for common use cases
- **Documentation** for each component
- **Examples** and usage patterns
- **Testing** for component consistency 