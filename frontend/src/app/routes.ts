export const routes = {
  home: '/', chat: '/chat', capture: '/capture', inventory: '/inventory',
  moveInfo: '/move-info', result: '/result', history: '/history', demo: '/dev/analysis',
} as const;

export type AppRoute = (typeof routes)[keyof typeof routes];

export function navigate(path: AppRoute) {
  window.history.pushState({}, '', path);
  window.dispatchEvent(new PopStateEvent('popstate'));
}
