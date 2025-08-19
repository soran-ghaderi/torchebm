# GitHub Issue Templates

This directory contains GitHub issue templates that provide a standardized way for users to report bugs, request features, and ask questions about TorchEBM.

## Available Templates

### 🐛 Bug Report (`bug_report.yml`)
For reporting bugs, errors, or unexpected behavior in TorchEBM.

**Key Features:**
- Prerequisites checklist to ensure quality reports
- Structured sections for reproduction steps and minimal examples
- System information collection (TorchEBM, PyTorch, Python versions)
- Device type selection (CPU/GPU/MPS)
- Error output formatting with syntax highlighting

### 🚀 Feature Request (`feature_request.yml`)
For suggesting new features, enhancements, or improvements.

**Key Features:**
- Problem and solution description sections
- Feature categorization (Energy Functions, Samplers, etc.)
- Priority assessment
- Example usage with code formatting
- Implementation contribution options

### ⚡ Performance Issue (`performance_issue.yml`)
For reporting performance problems, slowdowns, or optimization requests.

**Key Features:**
- Component identification (Energy Functions, Samplers, etc.)
- Benchmark code with timing examples
- System specifications collection
- Profiling results section
- Data scale assessment

### 📚 Documentation (`documentation.yml`)
For reporting documentation issues or suggesting improvements.

**Key Features:**
- Issue type classification (Missing, Incorrect, Unclear, etc.)
- Documentation section identification
- Target audience selection
- Content improvement suggestions
- Contribution willingness indicators

### ❓ Question (`question.yml`)
For asking questions about using TorchEBM.

**Key Features:**
- Question categorization
- Experience level assessment
- Context and current approach sections
- System information (when relevant)
- Urgency indicators

## Configuration

### `config.yml`
Controls the issue template chooser interface and provides helpful links.

**Current Configuration:**
- Disables blank issues (`blank_issues_enabled: false`)
- Provides contact links for documentation, discussions, and research papers
- Encourages users to check existing resources before creating issues

## Customization

### Updating Templates
To modify templates:

1. Edit the relevant `.yml` file
2. Test the template by creating a test issue
3. Adjust field requirements, options, or descriptions as needed

### Adding New Templates
To add a new template:

1. Create a new `.yml` file in this directory
2. Follow the GitHub issue forms syntax
3. Update this README to document the new template
4. Consider updating `config.yml` if additional contact links are needed

### Template Syntax
Templates use GitHub's issue forms syntax with these key elements:

- `name`: Template display name
- `description`: Brief description shown in template chooser
- `title`: Pre-filled issue title (can include placeholders)
- `labels`: Auto-applied labels
- `assignees`: Auto-assigned team members
- `body`: Form fields array

### Field Types
Available field types:
- `markdown`: Static text/instructions
- `textarea`: Multi-line text input
- `input`: Single-line text input
- `dropdown`: Selection menu
- `checkboxes`: Multiple choice checkboxes

## Labels

The templates use these labels (ensure they exist in your repository):

- `bug` - Bug reports
- `needs-triage` - New issues requiring review
- `enhancement` - Feature requests
- `feature-request` - Specific feature requests
- `performance` - Performance-related issues
- `needs-investigation` - Issues requiring deeper investigation
- `documentation` - Documentation-related issues
- `good-first-issue` - Suitable for new contributors
- `question` - User questions
- `help-wanted` - Community help requested

## Best Practices

### For Maintainers
1. **Review templates regularly** - Update based on common issue patterns
2. **Keep labels consistent** - Ensure all referenced labels exist
3. **Monitor template effectiveness** - Adjust based on issue quality
4. **Update contact links** - Keep documentation and discussion links current

### For Contributors
1. **Test templates** - Create test issues to verify functionality
2. **Keep descriptions clear** - Use helpful placeholder text
3. **Balance required/optional fields** - Don't overwhelm users
4. **Consider mobile users** - Keep forms reasonably sized

## Validation

Templates include validation rules:
- `required: true` - Field must be completed
- `options` - Predefined choices for dropdowns
- `render` - Syntax highlighting for code blocks

## Integration

These templates integrate with:
- GitHub's issue tracking system
- Repository labels and assignees
- Project boards (if configured)
- Automated workflows (if set up)

## Troubleshooting

Common issues and solutions:

1. **Template not showing** - Check YAML syntax and file naming
2. **Labels not applying** - Ensure labels exist in repository
3. **Validation errors** - Check required field configurations
4. **Contact links broken** - Update URLs in `config.yml`

## Resources

- [GitHub Issue Forms Documentation](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms)
- [GitHub Issue Templates Guide](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository)
- [YAML Syntax Reference](https://yaml.org/spec/1.2/spec.html)
