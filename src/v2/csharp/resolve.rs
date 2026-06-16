//! C# name resolution for the Add-Type dialect.
//!
//! C# scoping is lexical and static, so binding can be exact rather than
//! heuristic. [`resolve`] walks a [`CsUnit`] and produces a [`Resolved`] that
//! answers two questions renaming needs: what symbols are declared, and which
//! source spans reference a given symbol.
//!
//! The model:
//!
//! * Each scope hoists its own declarations, so a method may reference a field
//!   declared later in the same type.
//! * A root identifier (not after `.`/`::`) binds to the nearest enclosing
//!   declaration, so a local shadows a field and a parameter shadows a field.
//! * A member-access identifier (after `.`/`::`) binds by name to the unit's
//!   member declarations. Within a single file (the Add-Type case) that is
//!   accurate; the one ambiguity, two declared members sharing a name, is
//!   documented rather than guessed away.

use super::ast::{CsName, CsNode, CsNodeKind, CsUnit};
use crate::v2::span::Span;
use std::collections::{HashMap, HashSet};

/// What a declaration introduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclKind {
    /// A namespace.
    Namespace,
    /// A class, struct, interface, or enum.
    Type,
    /// A method.
    Method,
    /// A constructor.
    Ctor,
    /// A property.
    Property,
    /// An enum member.
    EnumMember,
    /// A field.
    Field,
    /// A local variable.
    Local,
    /// A parameter.
    Param,
}

impl DeclKind {
    /// Whether this kind can be reached through member access (`x.Name`), and
    /// so collects after-dot references by name.
    fn is_member(self) -> bool {
        matches!(
            self,
            DeclKind::Type
                | DeclKind::Method
                | DeclKind::Ctor
                | DeclKind::Property
                | DeclKind::Field
                | DeclKind::EnumMember
        )
    }
}

/// A declared symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decl {
    /// What it declares.
    pub kind: DeclKind,
    /// Its name.
    pub name: String,
    /// The span of the declared name.
    pub span: Span,
    /// The scope the declaration lives in.
    pub scope: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScopeKind {
    Unit,
    Namespace,
    Type,
    Method,
    Property,
    Block,
}

struct Scope {
    parent: Option<usize>,
    kind: ScopeKind,
    names: HashMap<String, usize>, // name -> decl id
}

struct RefSite {
    text: String,
    span: Span,
    after_dot: bool,
    receiver: Option<String>,
    scope: usize,
}

/// A member-access reference: the accessed name, its span, the receiver it
/// was reached through (when that receiver is a bare name), and the type whose
/// body the reference sits in (needed so `this.X` binds to the enclosing
/// type's `X`, not to a same-named member of another type).
#[derive(Debug, Clone)]
struct MemberRef {
    text: String,
    span: Span,
    receiver: Option<String>,
    enclosing_type: Option<String>,
}

/// The result of resolving a [`CsUnit`]: its declarations and the references
/// bound to them. Self-contained (holds spans and names, not the tree).
#[derive(Debug, Clone)]
pub struct Resolved {
    decls: Vec<Decl>,
    /// For each declaration id, the spans of the root references bound to it.
    root_refs: Vec<Vec<Span>>,
    /// All member-access (after-dot) reference sites, kept for name matching.
    member_refs: Vec<MemberRef>,
    /// For each scope id, the name of the nearest enclosing type, if any.
    type_of_scope: Vec<Option<String>>,
    /// type name -> base type/interface names, for inheritance-aware binding.
    parents: HashMap<String, Vec<String>>,
    /// owner type name -> the member names it directly declares, for an O(1)
    /// `type_declares` lookup instead of scanning every declaration.
    members: HashMap<String, HashSet<String>>,
}

impl Resolved {
    /// All declared symbols, in source order of discovery.
    pub fn symbols(&self) -> &[Decl] {
        &self.decls
    }

    /// The name of the type that encloses a declaration, if any. Lets a caller
    /// scope a rename to one type's members.
    pub fn enclosing_type(&self, decl_id: usize) -> Option<&str> {
        let scope = self.decls.get(decl_id)?.scope;
        self.type_of_scope.get(scope)?.as_deref()
    }

    /// Declaration ids whose name matches `name` (exact; C# is case-sensitive),
    /// optionally filtered by kind.
    pub fn find(&self, name: &str, kind: Option<DeclKind>) -> Vec<usize> {
        self.decls
            .iter()
            .enumerate()
            .filter(|(_, d)| d.name == name && kind.is_none_or(|k| d.kind == k))
            .map(|(i, _)| i)
            .collect()
    }

    /// The declaration whose name span exactly equals `span`, if any.
    pub fn decl_at(&self, span: Span) -> Option<usize> {
        self.decls.iter().position(|d| d.span == span)
    }

    /// A declaration by id.
    pub fn decl(&self, id: usize) -> Option<&Decl> {
        self.decls.get(id)
    }

    /// Every span that should change when this declaration is renamed: the
    /// declaration name itself, every root reference bound to it, and (for a
    /// member kind) every member access of the same name reached through a
    /// receiver that can only be this member: `this`, `base`, or the declaring
    /// type's name (a static use like `Type.Member`). A member access through
    /// some other receiver (`other.Member`, `s.Length`) is left out, since its
    /// type is unknown. Sorted by start, deduplicated.
    pub fn references_of(&self, decl_id: usize) -> Vec<Span> {
        let Some(decl) = self.decls.get(decl_id) else {
            return Vec::new();
        };
        let mut spans = vec![decl.span];
        if let Some(refs) = self.root_refs.get(decl_id) {
            spans.extend(refs.iter().copied());
        }
        if decl.kind.is_member() {
            let owner = self
                .type_of_scope
                .get(decl.scope)
                .and_then(|t| t.as_deref());
            for m in &self.member_refs {
                if m.text == decl.name && self.receiver_is_this_member(m, owner) {
                    spans.push(m.span);
                }
            }
        }
        if decl.kind == DeclKind::Type {
            // A constructor's name is the type name, and `new T(...)` names the
            // type too. Both are collected against the constructor declaration
            // (its name matches the type and it is a member kind), so fold in
            // every constructor owned by this type. This is what makes a type
            // rename reach `public T()` and `new T()`.
            for (id, other) in self.decls.iter().enumerate() {
                if other.kind == DeclKind::Ctor
                    && self
                        .type_of_scope
                        .get(other.scope)
                        .and_then(|t| t.as_deref())
                        == Some(decl.name.as_str())
                {
                    spans.push(other.span);
                    if let Some(refs) = self.root_refs.get(id) {
                        spans.extend(refs.iter().copied());
                    }
                    for m in &self.member_refs {
                        if m.text == other.name
                            && self.receiver_is_this_member(m, Some(decl.name.as_str()))
                        {
                            spans.push(m.span);
                        }
                    }
                }
            }
        }
        spans.sort_by_key(|s| (s.start, s.end));
        spans.dedup();
        spans
    }

    /// Whether a member-access receiver can only denote the current member's
    /// owner. `this` qualifies for a member of the enclosing type or one it
    /// inherits, binding to the most-derived declarer so a shadowed member
    /// renames against the shadowing type. `base.X` binds to the nearest
    /// ancestor that declares `X`, via the type-to-bases map; an external base
    /// (not present in the unit) ends the chain without binding. A bare-name
    /// receiver qualifies when it is the declaring type's name (static access),
    /// from anywhere. An unknown receiver (`None`) never qualifies.
    fn receiver_is_this_member(&self, m: &MemberRef, owner: Option<&str>) -> bool {
        let Some(owner) = owner else {
            return false;
        };
        let enclosing = m.enclosing_type.as_deref();
        match m.receiver.as_deref() {
            // `this.X` binds to the most-derived type on the enclosing type's
            // chain (itself included) that declares `X`. This makes a shadowed
            // member rename against the shadowing type, not the base.
            Some("this") => {
                enclosing.is_some_and(|e| self.nearest_decl_of(e, &m.text) == Some(owner))
            }
            // `base.X` skips the enclosing type: it binds to the most-derived
            // ancestor (strictly above) that declares `X`.
            Some("base") => enclosing
                .and_then(|e| self.parents.get(e))
                .into_iter()
                .flatten()
                .any(|b| self.nearest_decl_of(b, &m.text) == Some(owner)),
            // A named receiver matching the owner (e.g. a same-type static use).
            Some(r) => r == owner,
            None => false,
        }
    }

    /// The nearest type on `ty`'s chain (starting at `ty` itself) that declares
    /// a member named `member`, or `None` if no visible type on the chain does.
    /// Breadth-first so the most-derived declarer wins; the visited set guards
    /// cyclic bases.
    fn nearest_decl_of<'a>(&'a self, ty: &'a str, member: &str) -> Option<&'a str> {
        let mut queue = std::collections::VecDeque::from([ty]);
        let mut seen: HashSet<&str> = HashSet::new();
        while let Some(cur) = queue.pop_front() {
            if !seen.insert(cur) {
                continue;
            }
            if self.type_declares(cur, member) {
                return Some(cur);
            }
            if let Some(bases) = self.parents.get(cur) {
                for b in bases {
                    queue.push_back(b.as_str());
                }
            }
        }
        None
    }

    /// Whether type `ty` directly declares a member named `member`. Two borrow
    /// lookups against the index built in `into_resolved`: constant time, no
    /// per-call allocation.
    fn type_declares(&self, ty: &str, member: &str) -> bool {
        self.members
            .get(ty)
            .is_some_and(|names| names.contains(member))
    }
}

/// Resolves a parsed C# unit.
pub fn resolve(unit: &CsUnit) -> Resolved {
    let mut b = Builder {
        scopes: Vec::new(),
        decls: Vec::new(),
        refs: Vec::new(),
        type_of_scope: Vec::new(),
        parents: HashMap::new(),
    };
    let root = b.new_scope(None, ScopeKind::Unit);
    for c in &unit.root.children {
        b.build(c, root);
    }
    b.into_resolved()
}

struct Builder {
    scopes: Vec<Scope>,
    decls: Vec<Decl>,
    refs: Vec<RefSite>,
    type_of_scope: Vec<Option<String>>,
    /// type name -> its declared base type/interface names.
    parents: HashMap<String, Vec<String>>,
}

impl Builder {
    fn new_scope(&mut self, parent: Option<usize>, kind: ScopeKind) -> usize {
        let ty = match parent {
            Some(p) => self.type_of_scope[p].clone(),
            None => None,
        };
        self.scopes.push(Scope {
            parent,
            kind,
            names: HashMap::new(),
        });
        self.type_of_scope.push(ty);
        self.scopes.len() - 1
    }

    fn add(&mut self, scope: usize, kind: DeclKind, name: &CsName) {
        let id = self.decls.len();
        self.decls.push(Decl {
            kind,
            name: name.text.clone(),
            span: name.span,
            scope,
        });
        // First declaration of a name wins the slot; later same-name decls are
        // still recorded (for find/references) but do not shadow within a scope.
        self.scopes[scope]
            .names
            .entry(name.text.clone())
            .or_insert(id);
    }

    fn build(&mut self, node: &CsNode, scope: usize) {
        match &node.kind {
            CsNodeKind::Type { name, bases, .. } => {
                self.add(scope, DeclKind::Type, name);
                if !bases.is_empty() {
                    self.parents
                        .entry(name.text.clone())
                        .or_default()
                        .extend(bases.iter().map(|b| b.text.clone()));
                }
                let inner = self.new_scope(Some(scope), ScopeKind::Type);
                self.type_of_scope[inner] = Some(name.text.clone());
                self.build_children(node, inner);
            }
            CsNodeKind::Namespace(name) => {
                self.add(scope, DeclKind::Namespace, name);
                let inner = self.new_scope(Some(scope), ScopeKind::Namespace);
                self.build_children(node, inner);
            }
            CsNodeKind::Method(name) => {
                self.add(scope, DeclKind::Method, name);
                let inner = self.new_scope(Some(scope), ScopeKind::Method);
                self.build_children(node, inner);
            }
            CsNodeKind::Ctor(name) => {
                self.add(scope, DeclKind::Ctor, name);
                let inner = self.new_scope(Some(scope), ScopeKind::Method);
                self.build_children(node, inner);
            }
            CsNodeKind::Property(name) => {
                self.add(scope, DeclKind::Property, name);
                let inner = self.new_scope(Some(scope), ScopeKind::Property);
                self.build_children(node, inner);
            }
            CsNodeKind::Block => {
                let inner = self.new_scope(Some(scope), ScopeKind::Block);
                self.build_children(node, inner);
            }
            CsNodeKind::Param(name) => {
                self.add(scope, DeclKind::Param, name);
            }
            CsNodeKind::EnumMember(name) => {
                self.add(scope, DeclKind::EnumMember, name);
            }
            CsNodeKind::NameDecl(name) => {
                let kind = if self.scopes[scope].kind == ScopeKind::Type {
                    DeclKind::Field
                } else {
                    DeclKind::Local
                };
                self.add(scope, kind, name);
            }
            CsNodeKind::NameRef {
                name,
                after_dot,
                receiver,
            } => {
                self.refs.push(RefSite {
                    text: name.text.clone(),
                    span: name.span,
                    after_dot: *after_dot,
                    receiver: receiver.clone(),
                    scope,
                });
            }
            // Decl group, Attribute, Using, Error, Unit: no scope of their own.
            _ => self.build_children(node, scope),
        }
    }

    fn build_children(&mut self, node: &CsNode, scope: usize) {
        for c in &node.children {
            self.build(c, scope);
        }
    }

    fn lookup(&self, mut scope: usize, name: &str) -> Option<usize> {
        loop {
            if let Some(id) = self.scopes[scope].names.get(name) {
                return Some(*id);
            }
            scope = self.scopes[scope].parent?;
        }
    }

    fn into_resolved(self) -> Resolved {
        let mut root_refs = vec![Vec::new(); self.decls.len()];
        let mut member_refs = Vec::new();
        for r in &self.refs {
            if r.after_dot {
                member_refs.push(MemberRef {
                    text: r.text.clone(),
                    span: r.span,
                    receiver: r.receiver.clone(),
                    enclosing_type: self.type_of_scope[r.scope].clone(),
                });
            } else if let Some(id) = self.lookup(r.scope, &r.text) {
                root_refs[id].push(r.span);
            }
        }
        // Index the members each type declares, for inheritance lookups.
        let mut members: HashMap<String, HashSet<String>> = HashMap::new();
        for d in &self.decls {
            if matches!(
                d.kind,
                DeclKind::Field | DeclKind::Method | DeclKind::Property | DeclKind::EnumMember
            ) {
                if let Some(Some(owner)) = self.type_of_scope.get(d.scope) {
                    members
                        .entry(owner.clone())
                        .or_default()
                        .insert(d.name.clone());
                }
            }
        }
        Resolved {
            decls: self.decls,
            root_refs,
            member_refs,
            type_of_scope: self.type_of_scope,
            parents: self.parents,
            members,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::csharp::parser::cs_parse;

    fn resolve_src(src: &str) -> Resolved {
        resolve(&cs_parse(src, 0))
    }

    /// The set of texts referenced for the (first) declaration named `name`.
    fn ref_texts(src: &str, name: &str, kind: DeclKind) -> Vec<String> {
        let r = resolve_src(src);
        let ids = r.find(name, Some(kind));
        let id = *ids.first().expect("declaration not found");
        r.references_of(id)
            .into_iter()
            .map(|s| s.slice(src).to_string())
            .collect()
    }

    #[test]
    fn field_reference_resolves_bare_and_through_this() {
        let src = "class C { int count; void M() { count = count + this.count; } }";
        let refs = ref_texts(src, "count", DeclKind::Field);
        // declaration + three uses (two bare, one this.count)
        assert_eq!(refs.iter().filter(|t| *t == "count").count(), 4);
    }

    #[test]
    fn a_local_shadows_the_field_of_the_same_name() {
        // The bare `x` inside M binds to the local, not the field, so renaming
        // the field must NOT touch it.
        let src = "class C { int x; void M() { int x = 0; x = x + 1; } int Other() { return x; } }";
        let r = resolve_src(src);
        let field = *r.find("x", Some(DeclKind::Field)).first().unwrap();
        let local = *r.find("x", Some(DeclKind::Local)).first().unwrap();
        let field_refs = r.references_of(field);
        let local_refs = r.references_of(local);
        // Field: its declaration + the `return x` in Other (1 use). Not the
        // three x's inside M.
        assert_eq!(field_refs.len(), 2, "field refs: {field_refs:?}");
        // Local: its declaration + two uses in `x = x + 1`.
        assert_eq!(local_refs.len(), 3, "local refs: {local_refs:?}");
    }

    #[test]
    fn method_call_sites_resolve() {
        let src = "class C { void Helper() {} void M() { Helper(); this.Helper(); } }";
        let refs = ref_texts(src, "Helper", DeclKind::Method);
        // declaration + bare call + this.Helper
        assert_eq!(refs.iter().filter(|t| *t == "Helper").count(), 3);
    }

    #[test]
    fn parameter_references_resolve_within_the_method() {
        let src = "class C { int Add(int a, int b) { return a + b + a; } }";
        let refs = ref_texts(src, "a", DeclKind::Param);
        // declaration + two uses of `a`
        assert_eq!(refs.len(), 3);
    }

    #[test]
    fn type_references_include_constructors_and_new() {
        // X1 at the resolver level: a type collects its declaration, every
        // constructor name, and every `new T(...)` site.
        let src = "class Logger {\n public Logger() { }\n public Logger(int n) { }\n void M() { var a = new Logger(); var b = new Logger(5); }\n}";
        let r = resolve_src(src);
        let id = *r.find("Logger", Some(DeclKind::Type)).first().unwrap();
        let refs = r.references_of(id);
        assert_eq!(refs.len(), 5, "decl + 2 ctors + 2 new: {refs:?}");
        // No duplicate spans.
        let mut keys: Vec<_> = refs.iter().map(|s| (s.start, s.end)).collect();
        let total = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), total, "spans must be unique");
    }

    #[test]
    fn generic_argument_is_not_treated_as_a_base() {
        // `class C : Dictionary<string, List>` does not make C inherit List, so
        // a List member is not reached through C.
        let src = "class List { public int Count; }\nclass C : Dictionary<string, List> { public void U() { this.Count = 1; } }";
        let r = resolve_src(src);
        let count = *r.find("Count", Some(DeclKind::Field)).first().unwrap();
        assert_eq!(
            r.references_of(count).len(),
            1,
            "List.Count should be its declaration only, not bound through C"
        );

        // A real generic base still inherits: Base<int> contributes Base.
        let inherit =
            "class Base { public int X; }\nclass B : Base<int> { public void U() { this.X = 1; } }";
        let r = resolve_src(inherit);
        let x = *r.find("X", Some(DeclKind::Field)).first().unwrap();
        assert_eq!(r.references_of(x).len(), 2, "generic base still inherits");
    }

    #[test]
    fn inherited_members_bind_through_this_and_base() {
        // base.X and an inherited this.X bind to the base type's member, so a
        // rename of the base member reaches every inherited use.
        let src = "class A {\n public int Count;\n public void M() { }\n}\nclass B : A {\n public void Use() {\n  this.Count = 1;\n  base.Count = 2;\n  base.M();\n }\n}";
        let r = resolve_src(src);
        let count = *r.find("Count", Some(DeclKind::Field)).first().unwrap();
        assert_eq!(r.references_of(count).len(), 3, "decl + this + base");
        let m = *r.find("M", Some(DeclKind::Method)).first().unwrap();
        assert_eq!(r.references_of(m).len(), 2, "decl + base.M");
    }

    #[test]
    fn shadowed_member_binds_to_most_derived() {
        // B redeclares Count. `this.Count` in B binds to B's, `base.Count`
        // binds to A's.
        let via_this =
            "class A { public int Count; }\nclass B : A { public int Count; public void U() { this.Count = 1; } }";
        let r = resolve_src(via_this);
        let ids = r.find("Count", Some(DeclKind::Field));
        assert_eq!(ids.len(), 2);
        for id in ids {
            let refs = r.references_of(id);
            // A.Count is declared earlier in the source than B.Count.
            if refs[0].start < 25 {
                assert_eq!(refs.len(), 1, "A.Count: declaration only");
            } else {
                assert_eq!(refs.len(), 2, "B.Count: declaration + this.Count");
            }
        }

        let via_base =
            "class A { public int Count; }\nclass B : A { public int Count; public void U() { base.Count = 1; } }";
        let r = resolve_src(via_base);
        for id in r.find("Count", Some(DeclKind::Field)) {
            let refs = r.references_of(id);
            if refs[0].start < 25 {
                assert_eq!(refs.len(), 2, "A.Count: declaration + base.Count");
            } else {
                assert_eq!(refs.len(), 1, "B.Count: declaration only");
            }
        }
    }

    #[test]
    fn external_and_cyclic_bases_are_safe() {
        // A base type not present in the unit ends the chain (no false bind),
        // and a cyclic base list does not loop.
        let external = "class B : ExternalThing { public void U() { base.DoIt(); } }";
        let r = resolve_src(external);
        assert!(r.find("DoIt", Some(DeclKind::Method)).is_empty());

        let cyclic =
            "class A : B { public int X; }\nclass B : A { public void U() { this.X = 1; } }";
        let r = resolve_src(cyclic);
        // Must terminate; X resolves to the single declaration plus the this use.
        let x = *r.find("X", Some(DeclKind::Field)).first().unwrap();
        assert!(!r.references_of(x).is_empty());
    }

    #[test]
    fn this_references_do_not_leak_across_types() {
        // Both classes declare `Count` and use `this.Count`. Each field's
        // references must stay inside its own type: renaming A.Count must not
        // touch B's `this.Count`, and the other way round.
        let src = "class A { public int Count; void M() { this.Count = 1; } }\n\
                   class B { public int Count; void N() { this.Count = 2; } }";
        let r = resolve_src(src);
        for id in r.find("Count", Some(DeclKind::Field)) {
            let owner = r.enclosing_type(id).expect("field has a type").to_string();
            let refs = r.references_of(id);
            assert_eq!(refs.len(), 2, "decl + own this.Count for {owner}: {refs:?}");
            // Every span sits inside its own class's source half.
            let own_half = if owner == "A" { 0..59 } else { 59..src.len() };
            for s in &refs {
                assert!(
                    own_half.contains(&s.start),
                    "{owner}.Count ref at {s:?} leaked outside its class"
                );
            }
        }
        // Static access through the type name still binds from another type.
        let src2 = "class A { public static int Count; }\nclass B { void N() { A.Count = 5; } }";
        let r2 = resolve_src(src2);
        let id = *r2.find("Count", Some(DeclKind::Field)).first().unwrap();
        assert_eq!(r2.references_of(id).len(), 2, "decl + A.Count static use");
    }

    #[test]
    fn type_references_resolve() {
        let src = "class Widget { } class C { Widget w; Widget Make() { return new Widget(); } }";
        let refs = ref_texts(src, "Widget", DeclKind::Type);
        // declaration + field type + return type + `new Widget`
        assert_eq!(refs.iter().filter(|t| *t == "Widget").count(), 4);
    }
}
