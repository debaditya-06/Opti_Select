import { useLayoutEffect } from 'react';
import type { BreadcrumbItem } from '../types';

interface PageMetaOptions {
  title: string;
  description: string;
  canonicalPath: string;
  breadcrumbs?: BreadcrumbItem[];
}

export function usePageMeta({
  title,
  description,
  canonicalPath,
  breadcrumbs,
}: PageMetaOptions): void {
  useLayoutEffect(() => {
    document.title = title;

    let metaDesc = document.querySelector<HTMLMetaElement>('meta[name="description"]');
    if (!metaDesc) {
      metaDesc = document.createElement('meta');
      metaDesc.name = 'description';
      document.head.appendChild(metaDesc);
    }
    metaDesc.content = description;

    const origin = window.location.origin;
    let canonical = document.querySelector<HTMLLinkElement>('link[rel="canonical"]');
    if (!canonical) {
      canonical = document.createElement('link');
      canonical.rel = 'canonical';
      document.head.appendChild(canonical);
    }
    canonical.href = `${origin}${canonicalPath}`;

    const existingJsonLd = document.getElementById('breadcrumb-jsonld');
    if (breadcrumbs && breadcrumbs.length > 0) {
      const itemListElement = breadcrumbs.map((item, idx) => ({
        '@type': 'ListItem',
        position: idx + 1,
        name: item.label,
        item: `${origin}${item.path ?? canonicalPath}`,
      }));

      const payload = {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement,
      };

      const script = existingJsonLd ?? document.createElement('script');
      script.id = 'breadcrumb-jsonld';
      script.setAttribute('type', 'application/ld+json');
      script.textContent = JSON.stringify(payload);
      if (!existingJsonLd) {
        document.head.appendChild(script);
      }
    } else if (existingJsonLd) {
      existingJsonLd.remove();
    }
  }, [title, description, canonicalPath, breadcrumbs]);
}
